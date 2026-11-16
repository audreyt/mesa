/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "hk-apple9-test.h"
#include "asahi/lib/agx_apple9.h"
#include "asahi/lib/agx_apple9_layout.h"
#include "asahi/genxml/agx_pack.h"
#include "hk_descriptor_set_layout.h"
#include "nir_builder.h"
#include "vk_pipeline.h"
#include "vk_shader.h"

TEST(Apple9Descriptors, LinearTextureDepthDoesNotAliasStride)
{
   for (bool linear : {false, true}) {
      SCOPED_TRACE(linear);
      struct agx_texture_packed descriptor;
      struct AGX_TEXTURE cfg = {AGX_TEXTURE_header};
      cfg.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
      cfg.layout = linear ? AGX_LAYOUT_LINEAR : AGX_LAYOUT_TWIDDLED;
      cfg.width = 1440;
      cfg.height = 900;
      cfg.address = 0x4000;
      cfg.extended = linear;
      if (linear) {
         cfg.stride = 5744;
         cfg.depth_linear = 1;
      } else {
         cfg.depth = 4;
      }
      AGX_TEXTURE_pack(reinterpret_cast<uint32_t *>(&descriptor), &cfg);
      std::array<uint8_t, AGX_APPLE9_TEXTURE_TABLE_STRIDE> translated, direct;
      ASSERT_TRUE(agx_apple9_translate_texture(translated.data(), &descriptor));
      ASSERT_TRUE(agx_apple9_pack_texture(direct.data(), &cfg));
      EXPECT_EQ(translated, direct);
   }
}

TEST(Apple9Descriptors, ConstantSelectionReadsTheLiveDescriptor)
{
   auto *layout = static_cast<hk_descriptor_set_layout *>(
      calloc(1, sizeof(hk_descriptor_set_layout) +
                   sizeof(hk_descriptor_set_binding_layout)));
   layout->binding_count = 1;
   layout->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   layout->binding[0].array_size = 4;
   layout->binding[0].stride = 16;
   vk_descriptor_set_layout *sets[] = {&layout->vk};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "constant_descriptor_selection");
   nir_def *idx = nir_vulkan_resource_index(
      &b, 4, 32, nir_imm_int(&b, 2), .desc_set = 0, .binding = 0,
      .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      &b, 4, 32, idx, .desc_type = nir_descriptor_type_storage_buffer);
   auto *ptr = nir_build_deref_cast(&b, desc, nir_var_mem_ssbo,
                                   glsl_uint_type(), 4);
   nir_def *value = nir_load_deref(&b, ptr);
   nir_store_deref(&b, nir_build_deref_ptr_as_array(&b, ptr, nir_imm_int(&b, 1)),
                   value, 1);
   hk_apple9_test_lower_descriptors(
      b.shader, &vk_robustness_disabled, 1, sets);
   nir_lower_explicit_io(b.shader, nir_var_mem_ssbo,
                        nir_address_format_64bit_global_32bit_offset);
   ASSERT_TRUE(hk_apple9_test_prepare_native_nir(b.shader));
   /* Supply set zero through the actual native root UBO adapter. */
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_ubo)
            continue;
         ASSERT_EQ(nir_src_as_uint(intr->src[0]), 0u);
         ASSERT_EQ(nir_src_as_uint(intr->src[1]), 8u);
         b.cursor = nir_before_instr(instr);
         nir_def_replace(&intr->def, nir_imm_int64(&b, 0x4000));
      }
   }
   nir_opt_constant_folding(b.shader);
   nir_opt_dce(b.shader);
   unsigned descriptors = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_global_constant &&
             intr->intrinsic != nir_intrinsic_load_global_constant_offset)
            continue;
         ++descriptors;
         ASSERT_TRUE(nir_src_is_const(intr->src[0]));
         uint64_t address = nir_src_as_uint(intr->src[0]);
         if (intr->intrinsic == nir_intrinsic_load_global_constant_offset) {
            ASSERT_TRUE(nir_src_is_const(intr->src[1]));
            address += nir_src_as_uint(intr->src[1]);
         }
         EXPECT_EQ(address, 0x4020u);
         EXPECT_EQ(intr->def.bit_size * intr->def.num_components, 128u);
      }
   }
   EXPECT_EQ(descriptors, 1u);
   nir_validate_shader(b.shader, "live constant descriptor selection");
   ralloc_free(b.shader);
   free(layout);
}

/* Walk the actual address graph, including loads of set and buffer pointers. */
struct dependency {
   nir_def *target;
   std::unordered_set<nir_instr *> visited;
};

static bool
independent(nir_src *src, void *data)
{
   auto *dep = static_cast<dependency *>(data);
   if (src->ssa == dep->target)
      return false;
   nir_instr *instr = nir_def_instr(src->ssa);
   return !dep->visited.insert(instr).second ||
          nir_foreach_src(instr, independent, data);
}

static bool
depends_on(nir_def *value, nir_def *target)
{
   dependency dep = {target, {}};
   nir_src src = nir_src_for_ssa(value);
   return !independent(&src, &dep);
}

class RootDescriptors : public ::testing::Test {
protected:
   hk_descriptor_set_layout *layouts[2] = {};
   vk_descriptor_set_layout *sets[2] = {};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "root_descriptors");
   nir_def *index = nullptr;
   vk_pipeline_robustness_state rs = {};

   void SetUp() override
   {
      for (unsigned s = 0; s < 2; ++s) {
         layouts[s] = static_cast<hk_descriptor_set_layout *>(calloc(
            1, sizeof(hk_descriptor_set_layout) +
                  2 * sizeof(hk_descriptor_set_binding_layout)));
         layouts[s]->binding_count = 2;
         sets[s] = &layouts[s]->vk;
      }
      layouts[0]->binding[0] = {
         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, 4, 0, 16, 0, nullptr};
      layouts[0]->binding[1] = {
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT, 300, 64, 16, 0,
         nullptr};
      layouts[1]->binding[0] = {
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 0, 4, 0, 16, 0, nullptr};
      layouts[1]->vk.dynamic_descriptor_count = 4;
      layouts[1]->binding[1] = {
         VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK, 0, 256, 64, 1, 0, nullptr};
      rs = vk_robustness_disabled;
      b.shader->info.workgroup_size[0] = 32;
      index = nir_load_subgroup_invocation(&b);
   }

   void TearDown() override
   {
      ralloc_free(b.shader);
      for (auto *layout : layouts)
         free(layout);
   }

   void storage_layout(VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
   {
      layouts[0]->binding[0] = {
         type, 0, 1, 0, hk_apple9_test_get_storage_layout().stride, 0, nullptr};
      layouts[0]->binding[1].offset = layouts[0]->binding[0].stride;
   }

   nir_deref_instr *buffer(unsigned set, unsigned binding, bool reindex = true)
   {
      bool ssbo = layouts[set]->binding[binding].type ==
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
                  layouts[set]->binding[binding].type ==
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      nir_descriptor_type type = ssbo ? nir_descriptor_type_storage_buffer
                                     : nir_descriptor_type_uniform_buffer;
      nir_def *resource = nir_vulkan_resource_index(
         &b, 4, 32, index, .desc_set = set, .binding = binding,
         .desc_type = type);
      if (reindex)
         resource = nir_vulkan_resource_reindex(
            &b, 32, resource, index, .desc_type = type);
      nir_def *desc = nir_load_vulkan_descriptor(
         &b, 4, 32, resource, .desc_type = type);
      auto mode = ssbo ? nir_var_mem_ssbo : nir_var_mem_ubo;
      nir_deref_instr *ptr = nir_build_deref_cast(
         &b, desc, mode, glsl_uint_type(), 4);
      return nir_build_deref_ptr_as_array(&b, ptr, index);
   }

   void shader()
   {
      nir_def *sum = nir_load_deref(&b, buffer(0, 0));
      auto *output = buffer(0, 1);
      sum = nir_iadd(&b, sum, nir_load_deref(&b, output));
      sum = nir_iadd(&b, sum, nir_load_deref(&b, buffer(1, 0)));
      sum = nir_iadd(&b, sum, nir_load_deref(&b, buffer(1, 1, false)));
      sum = nir_iadd(&b, sum, nir_load_push_constant(
         &b, 1, 32, nir_imul_imm(&b, index, 4), .base = 0, .range = 256));
      nir_store_deref(&b, output, sum, 1);
   }

   void lower(bool robust)
   {
      if (robust) {
         rs.storage_buffers =
            VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT;
         rs.uniform_buffers =
            VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT;
      }
      nir_validate_shader(b.shader, "before buffer descriptors");
      hk_apple9_test_lower_nir(b.shader, &rs, 2, sets);
      ASSERT_TRUE(hk_apple9_test_prepare_native_nir(b.shader));
      nir_opt_constant_folding(b.shader);
      nir_opt_copy_prop(b.shader);
      nir_opt_dce(b.shader);
      nir_validate_shader(b.shader, "after native root descriptors");
   }

   unsigned alu_count(nir_op op)
   {
      unsigned count = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               ++count;
         }
      }
      return count;
   }

   void check_root()
   {
      std::vector<nir_def *> roots;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            auto *intr = nir_instr_as_intrinsic(instr);
            EXPECT_NE(intr->intrinsic, nir_intrinsic_vulkan_resource_index);
            EXPECT_NE(intr->intrinsic, nir_intrinsic_vulkan_resource_reindex);
            EXPECT_NE(intr->intrinsic, nir_intrinsic_load_vulkan_descriptor);
            EXPECT_NE(intr->intrinsic, nir_intrinsic_load_root_agx);
            EXPECT_NE(intr->intrinsic, nir_intrinsic_load_descriptor_set_agx);
            if (intr->intrinsic == nir_intrinsic_load_ubo) {
               EXPECT_EQ(nir_src_as_uint(intr->src[0]), 0u);
               roots.push_back(&intr->def);
            }
         }
      }
      ASSERT_FALSE(roots.empty());
      unsigned stores = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            auto *intr = nir_instr_as_intrinsic(instr);
            bool store = intr->intrinsic == nir_intrinsic_store_global;
            if (!store && intr->intrinsic != nir_intrinsic_load_global &&
                intr->intrinsic != nir_intrinsic_load_global_constant &&
                intr->intrinsic != nir_intrinsic_load_global_constant_offset &&
                intr->intrinsic != nir_intrinsic_load_global_constant_bounded)
               continue;
            nir_def *addr = intr->src[store ? 1 : 0].ssa;
            EXPECT_EQ(addr->bit_size, 64);
            bool rooted = false;
            for (auto *root : roots)
               rooted |= depends_on(addr, root);
            EXPECT_TRUE(rooted);
            stores += store;
         }
      }
      EXPECT_EQ(stores, 1u);
   }
};

TEST_F(RootDescriptors, TwoSetsUnboundedGlobals)
{
   shader();
   lower(false);
   check_root();
   EXPECT_EQ(alu_count(nir_op_umin), 0u);
   EXPECT_EQ(alu_count(nir_op_bcsel), 0u);
}

TEST_F(RootDescriptors, RootLoadsKeepEffectiveRangeAndAlignment)
{
   nir_def *resource = nir_vulkan_resource_index(
      &b, 4, 32, nir_imm_int(&b, 0), .desc_set = 0, .binding = 1,
      .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *descriptor = nir_load_vulkan_descriptor(
      &b, 4, 32, resource, .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *binding = descriptor;
   nir_def *push = nir_load_push_constant(
      &b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 16,
      .align_mul = 256, .align_offset = 0);
   nir_def *group = nir_channel(&b, nir_load_base_workgroup_id(&b, 32), 0);
   nir_def *size = nir_get_ssbo_size(&b, 32, binding);
   nir_store_ssbo(&b, nir_iadd(&b, push, nir_iadd(&b, group, size)),
                  binding, nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   std::set<unsigned> widths;
   std::set<unsigned> root_offsets;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_ubo)
            continue;
         ASSERT_TRUE(nir_src_is_const(intr->src[1]));
         unsigned address = nir_src_as_uint(intr->src[1]);
         ASSERT_GT(nir_intrinsic_range(intr), 0u);
         EXPECT_EQ(nir_intrinsic_range_base(intr), address);
         EXPECT_EQ(nir_intrinsic_align_offset(intr),
                   address & (nir_intrinsic_align_mul(intr) - 1));
         EXPECT_EQ(nir_intrinsic_range(intr), 8u);
         b.cursor = nir_before_instr(instr);
         nir_def_replace(&intr->def,
            nir_imm_int64(&b, address == 0 ? 0x1080 : 0x2000));
         root_offsets.insert(address);
      }
   }
   EXPECT_EQ(root_offsets, (std::set<unsigned>{0, 8}));
   nir_opt_constant_folding(b.shader);
   nir_opt_dce(b.shader);
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_global_constant)
            continue;
         ASSERT_TRUE(nir_src_is_const(intr->src[0]));
         unsigned align = nir_intrinsic_align_mul(intr);
         EXPECT_EQ(nir_src_as_uint(intr->src[0]) & (align - 1),
                   nir_intrinsic_align_offset(intr));
         widths.insert(intr->def.num_components * intr->def.bit_size / 8);
      }
   }
   EXPECT_EQ(widths, (std::set<unsigned>{4, 12, 16}));
   nir_validate_shader(b.shader, "root field metadata");
}

TEST_F(RootDescriptors, RobustBoundsAndArrayClamps)
{
   shader();
   lower(true);
   check_root();
   EXPECT_GE(alu_count(nir_op_bcsel), 4u);
   std::vector<nir_def *> fixed_clamps;
   unsigned runtime_clamps = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         auto *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_umin || !nir_src_is_const(alu->src[1].src))
            continue;
         unsigned bound = nir_src_as_uint(alu->src[1].src);
         if (bound == 3) {
            EXPECT_TRUE(depends_on(alu->src[0].src.ssa, index));
            fixed_clamps.push_back(&alu->def);
         }
         if (bound == 299) {
            EXPECT_TRUE(depends_on(alu->src[0].src.ssa, index));
            ++runtime_clamps;
         }
      }
   }
   unsigned fixed_descriptors = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_global_constant ||
             intr->def.bit_size * intr->def.num_components != 128)
            continue;
         bool clamped = false;
         for (auto *clamp : fixed_clamps)
            clamped |= depends_on(intr->src[0].ssa, clamp);
         fixed_descriptors += clamped;
      }
   }
   /* CSE may share the clamp, but both descriptor addresses must use it. */
   EXPECT_EQ(fixed_descriptors, 2u); /* UBO[4] and dynamic SSBO[4] */
   EXPECT_EQ(runtime_clamps, 1u); /* runtime SSBO, not a 64-slot table */
}

TEST_F(RootDescriptors, RuntimeArray300RemainsDynamic)
{
   shader();
   lower(false);
   check_root();
   unsigned dynamic_addresses = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_global_constant &&
             depends_on(intr->src[0].ssa, index))
            ++dynamic_addresses;
      }
   }
   EXPECT_GE(dynamic_addresses, 2u);
}

TEST_F(RootDescriptors, NativeCompilerAcceptsLiveRootDescriptorGraph)
{
   shader();
   lower(true);
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   if (!ok)
      nir_print_shader(b.shader, stderr);
   ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_EQ(reason, nullptr);
   EXPECT_GT(compiled.info.binary_size, 0u);
   EXPECT_NE(compiled.binary, nullptr);
   ASSERT_EQ(profile.resource_binding_count, 1);
   EXPECT_EQ(profile.resource_binding[0], 0u);
   EXPECT_EQ(profile.resource_kind[0], AGX_APPLE9_COMPUTE_RESOURCE_UBO);
   free(compiled.binary);
}

TEST_F(RootDescriptors, R32ImageStoreCompilesWithoutArtificialBufferOutput)
{
   storage_layout();
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
      "storage_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_image_deref_store(
      &b, deref, nir_pad_vec4(&b, nir_vec2(&b, index, nir_imm_int(&b, 0))),
      nir_undef(&b, 1, 32),
      nir_vec4(&b, nir_iadd_imm(&b, index, 17), nir_imm_int(&b, 0),
                nir_imm_int(&b, 0), nir_imm_int(&b, 0)),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_R32_UINT, .access = ACCESS_NON_READABLE,
      .src_type = nir_type_uint32);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 0u);
   EXPECT_TRUE(profile.writes_global);
   free(compiled.binary);
}

TEST_F(RootDescriptors, R32ImageLoadCompilesSuccessfully)
{
   storage_layout();
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
      "storage_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_def *loaded = nir_image_deref_load(
      &b, 4, 32, deref,
      nir_pad_vec4(&b, nir_vec2(&b, index, nir_imm_int(&b, 0))),
      nir_undef(&b, 1, 32), nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_R32_UINT, .access = ACCESS_NON_WRITEABLE,
      .dest_type = nir_type_uint32);
   /* Use the loaded value in an SSBO store so the compiler doesn't DCE it */
   layouts[0]->binding[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   layouts[0]->binding[1].stride = 16;
   nir_store_ssbo(&b, nir_channel(&b, loaded, 0), nir_imm_int(&b, 1),
                  nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
}

TEST_F(RootDescriptors, Image1DArrayChecksLayerForLoadsAndStores)
{
   storage_layout();
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_1D, true, GLSL_TYPE_UINT), "array_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_def *coord = nir_pad_vec4(&b, nir_vec2(&b, nir_imm_int(&b, 0), index));
   nir_image_deref_store(
      &b, deref, coord, nir_undef(&b, 1, 32), nir_imm_ivec4(&b, 17, 0, 0, 0),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_1D,
      .image_array = true, .format = PIPE_FORMAT_R32_UINT,
      .access = ACCESS_NON_READABLE, .src_type = nir_type_uint32);
   nir_def *loaded = nir_image_deref_load(
      &b, 4, 32, deref, coord, nir_undef(&b, 1, 32), nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_1D, .image_array = true,
      .format = PIPE_FORMAT_R32_UINT, .access = ACCESS_NON_WRITEABLE,
      .dest_type = nir_type_uint32);
   layouts[0]->binding[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   layouts[0]->binding[1].stride = 16;
   nir_store_ssbo(&b, nir_channel(&b, loaded, 0), nir_imm_int(&b, 1),
                  nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   std::vector<nir_def *> layer_guards;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         auto *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_ult || alu->def.num_components != 1 ||
             nir_src_is_const(alu->src[1].src))
            continue;
         nir_scalar layer = nir_scalar_resolved(
            alu->src[0].src.ssa, alu->src[0].swizzle[0]);
         if (layer.def == index && layer.comp == 0)
            layer_guards.push_back(&alu->def);
      }
   }
   unsigned reads = 0, writes = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_global)
            ++reads;
         else if (intr->intrinsic == nir_intrinsic_store_global)
            ++writes;
         else
            continue;
         bool guarded = false;
         for (auto *node = &block->cf_node; node->parent; node = node->parent) {
            if (node->parent->type != nir_cf_node_if)
               continue;
            auto *nif = nir_cf_node_as_if(node->parent);
            auto *first = node;
            while (nir_cf_node_prev(first))
               first = nir_cf_node_prev(first);
            if (first != &nir_if_first_then_block(nif)->cf_node)
               continue;
            nir_def *condition = nif->condition.ssa;
            for (auto *guard : layer_guards) {
               guarded |= condition == guard;
               auto *condition_instr = nir_def_instr(condition);
               if (condition_instr->type != nir_instr_type_alu)
                  continue;
               auto *alu = nir_instr_as_alu(condition_instr);
               if (alu->op == nir_op_iand)
                  guarded |= alu->src[0].src.ssa == guard ||
                             alu->src[1].src.ssa == guard;
            }
         }
         EXPECT_TRUE(guarded);
      }
   }
   EXPECT_EQ(reads, 1u);
   EXPECT_EQ(writes, 1u);
}

TEST_F(RootDescriptors, R32TexelBufferLoadCompiles)
{
   storage_layout(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_BUF, false, GLSL_TYPE_UINT), "buffer_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;
   nir_def *loaded = nir_image_deref_load(
      &b, 4, 32, &nir_build_deref_var(&b, var)->def, nir_pad_vec4(&b, index),
      nir_undef(&b, 1, 32), nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_BUF, .format = PIPE_FORMAT_R32_UINT,
      .access = ACCESS_NON_WRITEABLE, .dest_type = nir_type_uint32);
   layouts[0]->binding[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   layouts[0]->binding[1].stride = 16;
   nir_store_ssbo(&b, nir_channel(&b, loaded, 0), nir_imm_int(&b, 1),
                  nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   free(compiled.binary);
}

TEST_F(RootDescriptors, FormatlessStoreUsesDescriptorFormatAndTexelStride)
{
   storage_layout();
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
      "formatless_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_NONE;
   nir_image_deref_store(
      &b, &nir_build_deref_var(&b, var)->def,
      nir_pad_vec4(&b, nir_vec2(&b, index, nir_imm_int(&b, 0))),
      nir_undef(&b, 1, 32),
      nir_imm_ivec4(&b, 0x12345678, 0x23456789, 0x3456789a, 0x456789ab),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_NONE, .access = ACCESS_NON_READABLE,
      .src_type = nir_type_uint32);

   ASSERT_NO_FATAL_FAILURE(lower(false));
   const auto layout = hk_apple9_test_get_storage_layout();
   const uint64_t set_address = 0x2000;
   nir_shader *probe = nir_shader_clone(b.shader, b.shader);
   nir_builder inspect = nir_builder_create(nir_shader_get_entrypoint(probe));
   nir_foreach_block(block, inspect.impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_ubo)
            continue;
         ASSERT_TRUE(nir_src_is_const(intr->src[0]));
         ASSERT_TRUE(nir_src_is_const(intr->src[1]));
         ASSERT_EQ(nir_src_as_uint(intr->src[0]), 0u);
         ASSERT_EQ(nir_src_as_uint(intr->src[1]), 8u);
         inspect.cursor = nir_before_instr(instr);
         nir_def_replace(&intr->def, nir_imm_int64(&inspect, set_address));
      }
   }
   nir_opt_constant_folding(probe);
   nir_def *texel_size = nullptr, *components = nullptr, *encoding = nullptr;
   std::vector<nir_intrinsic_instr *> stores;
   bool wide_store = false;
   nir_foreach_block(block, inspect.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_global_constant &&
             intr->def.bit_size == 32 &&
             nir_src_is_const(intr->src[0])) {
            uint64_t offset = nir_src_as_uint(intr->src[0]) - set_address;
            if (offset == layout.texel_size_offset)
               texel_size = &intr->def;
            if (offset == layout.components_offset)
               components = &intr->def;
            if (offset == layout.store_encode_offset)
               encoding = &intr->def;
         }
         /* Exclude assertion/printf stores emitted by the layout helper. */
         if (intr->intrinsic == nir_intrinsic_store_global &&
             (nir_intrinsic_access(intr) & ACCESS_NON_READABLE)) {
            stores.push_back(intr);
            wide_store |= intr->src[0].ssa->bit_size == 32;
         }
      }
   }
   ASSERT_NE(texel_size, nullptr);
   ASSERT_NE(components, nullptr);
   ASSERT_NE(encoding, nullptr);
   ASSERT_FALSE(stores.empty());
   EXPECT_TRUE(wide_store);
   for (auto *store : stores) {
      EXPECT_TRUE(depends_on(store->src[1].ssa, texel_size));
      EXPECT_TRUE(depends_on(store->src[0].ssa, components));
   }
   bool runtime_encoding = false;
   nir_foreach_block(block, inspect.impl) {
      if (auto *nif = nir_block_get_following_if(block))
         runtime_encoding |= depends_on(nif->condition.ssa, encoding);
   }
   EXPECT_TRUE(runtime_encoding);
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 0u);
   EXPECT_TRUE(profile.writes_global);
   free(compiled.binary);
}

TEST_F(RootDescriptors, FormatlessTexelBufferStoreCompiles)
{
   storage_layout(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_BUF, false, GLSL_TYPE_UINT),
      "formatless_buffer");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_NONE;
   nir_image_deref_store(
      &b, &nir_build_deref_var(&b, var)->def, nir_pad_vec4(&b, index),
      nir_undef(&b, 1, 32),
      nir_imm_ivec4(&b, 0x12345678, 0x23456789, 0, 0),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_BUF,
      .format = PIPE_FORMAT_NONE, .access = ACCESS_NON_READABLE,
      .src_type = nir_type_uint32);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_TRUE(profile.writes_global);
   free(compiled.binary);
}

TEST_F(RootDescriptors, ImageSizeCompilesSuccessfully)
{
   storage_layout();
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
      "storage_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_def *size = nir_image_deref_size(
      &b, 2, 32, deref, nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
      .format = PIPE_FORMAT_R32_UINT);
   /* Use the size in an SSBO store */
   layouts[0]->binding[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   layouts[0]->binding[1].stride = 16;
   nir_store_ssbo(&b, nir_channel(&b, size, 0), nir_imm_int(&b, 1),
                  nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
}

TEST_F(RootDescriptors, RGBA32ImageStoreCompilesSuccessfully)
{
   storage_layout();
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
      "storage_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32G32B32A32_UINT;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_image_deref_store(
      &b, deref, nir_pad_vec4(&b, nir_vec2(&b, index, nir_imm_int(&b, 0))),
      nir_undef(&b, 1, 32),
      nir_vec4(&b, nir_iadd_imm(&b, index, 17), nir_imm_int(&b, 0),
                nir_imm_int(&b, 0), nir_imm_int(&b, 0)),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_R32G32B32A32_UINT, .access = ACCESS_NON_READABLE,
      .src_type = nir_type_uint32);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 0u);
   EXPECT_TRUE(profile.writes_global);
   free(compiled.binary);
}

TEST_F(RootDescriptors, TextureDerefsKeepConstantSelection)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   /* Native combined records have one 128-byte image and sampler part. */
   layouts[0]->binding[0].stride = 256;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
      "sampled_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_def *coord = nir_imm_vec2(&b, 0.5, 0.5);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(coord);
   tex->src[1].src_type = nir_tex_src_texture_deref;
   tex->src[1].src = nir_src_for_ssa(deref);
   tex->src[2].src_type = nir_tex_src_sampler_deref;
   tex->src[2].src = nir_src_for_ssa(deref);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   ASSERT_TRUE(hk_apple9_test_lower_descriptors(b.shader, &rs, 2, sets));
   nir_opt_constant_folding(b.shader);
   EXPECT_EQ(nir_tex_instr_src_index(tex, nir_tex_src_texture_deref), -1);
   EXPECT_EQ(nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref), -1);
   EXPECT_EQ(tex->texture_index, 0u);
   EXPECT_EQ(tex->sampler_index, 0u);
   int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   int sampler = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   ASSERT_GE(texture, 0);
   ASSERT_GE(sampler, 0);
   auto *image = nir_src_as_intrinsic(tex->src[texture].src);
   auto *sample = nir_src_as_intrinsic(tex->src[sampler].src);
   ASSERT_EQ(image->intrinsic, nir_intrinsic_bindless_image_agx);
   ASSERT_EQ(sample->intrinsic, nir_intrinsic_bindless_sampler_agx);
   EXPECT_EQ(nir_intrinsic_desc_set(image), 0u);
   EXPECT_EQ(nir_intrinsic_desc_set(sample), 0u);
   EXPECT_EQ(nir_src_as_uint(image->src[0]), 0u);
   EXPECT_EQ(nir_src_as_uint(sample->src[0]), 128u);
}

TEST_F(RootDescriptors, UniformTexelBufferUsesNativeTextureTable)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
   layouts[0]->binding[0].stride = 128;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_BUF, false, false, GLSL_TYPE_FLOAT),
      "texel_buffer");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_BUF;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 1;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(nir_imm_int(&b, 17));
   tex->src[1].src_type = nir_tex_src_texture_deref;
   tex->src[1].src = nir_src_for_ssa(deref);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   ASSERT_TRUE(hk_apple9_test_lower_descriptors(b.shader, &rs, 2, sets));
   nir_opt_constant_folding(b.shader);

   EXPECT_EQ(nir_tex_instr_src_index(tex, nir_tex_src_texture_deref), -1);
   int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   ASSERT_GE(texture, 0);
   auto *handle = nir_src_as_intrinsic(tex->src[texture].src);
   EXPECT_EQ(handle->intrinsic, nir_intrinsic_bindless_image_agx);
   EXPECT_EQ(nir_intrinsic_desc_set(handle), 0u);
   EXPECT_EQ(nir_src_as_uint(handle->src[0]), 0u);
   EXPECT_EQ(nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle), -1);
}
