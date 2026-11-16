/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include "hk-apple9-test.h"
#include "asahi/compiler/agx_compile_apple9.h"
#include "asahi/lib/agx_apple9.h"
#include "asahi/lib/agx_nir_apple9_tilebuffer.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "hk_descriptor_set_layout.h"
#include "asahi/genxml/agx_pack.h"
#include "nir_builder.h"
#include "vk_meta_private.h"
#include "vk_pipeline.h"
#include "vk_shader.h"
#include "vk_graphics_state.h"
#include "vk_format.h"
#include "util/format/u_format.h"

static void
expect_packed_float(nir_src value, uint32_t expected)
{
   if (nir_src_is_const(value)) {
      EXPECT_EQ(nir_src_as_uint(value), expected);
      return;
   }

   /* Observe native packing through the independent CPU format reference.
    * The shader compiled above still contains the real hardware operation. */
   nir_instr *instr = nir_def_instr(value.ssa);
   ASSERT_EQ(instr->type, nir_instr_type_intrinsic);
   auto *pack = nir_instr_as_intrinsic(instr);
   ASSERT_EQ(pack->intrinsic, nir_intrinsic_pack_format_agx);
   ASSERT_EQ(nir_intrinsic_format(pack), PIPE_FORMAT_R11G11B10_FLOAT);
   ASSERT_TRUE(nir_src_is_const(pack->src[0]));
   ASSERT_EQ(pack->src[0].ssa->num_components, 3);
   const nir_const_value *rgb = nir_src_as_const_value(pack->src[0]);
   const float color[4] = {rgb[0].f32, rgb[1].f32, rgb[2].f32, 1.0f};
   uint32_t actual;
   util_format_pack_rgba(PIPE_FORMAT_R11G11B10_FLOAT, &actual, color, 1);
   EXPECT_EQ(actual, expected);
}

class Apple9MSAA : public ::testing::Test {
protected:
   void SetUp() override { glsl_type_singleton_init_or_ref(); }
   void TearDown() override { glsl_type_singleton_decref(); }
};

TEST_F(Apple9MSAA, LateMemoryDepthForcesNeverFailureWithoutFeedback)
{
   for (bool memory : {false, true})
   for (bool feedback : {false, true})
   for (bool early : {false, true})
   for (bool explicit_depth : {false, true})
   for (bool never : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << memory << feedback << early << explicit_depth << never);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "late depth side effects");
      b.shader->info.fs.early_fragment_tests = early;
      b.shader->info.num_ssbos = memory ? 1 : 0;
      b.shader->info.num_ubos = 1;
      if (memory)
         nir_store_ssbo(&b, nir_imm_int(&b, 0xf00d), nir_imm_int(&b, 0),
                        nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
      if (explicit_depth)
         nir_store_output(&b, nir_imm_float(&b, 0.875), nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = FRAG_RESULT_DEPTH, .num_slots = 1});

      hk_apple9_test_late_zs(b.shader, feedback);
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            auto *intr = nir_instr_as_intrinsic(instr);
            b.cursor = nir_before_instr(instr);
            if (intr->intrinsic == nir_intrinsic_load_frag_coord_z) {
               nir_def_replace(&intr->def, nir_imm_float(&b, 0.5));
            } else if (intr->intrinsic == nir_intrinsic_load_ubo) {
               ASSERT_EQ(nir_src_as_uint(intr->src[0]), 0u);
               if (intr->def.bit_size == 16) {
                  ASSERT_EQ(intr->def.num_components, 1);
                  nir_def_replace(&intr->def, nir_imm_intN_t(&b, never, 16));
               } else {
                  ASSERT_EQ(intr->def.bit_size, 32);
                  ASSERT_EQ(intr->def.num_components, 2);
                  nir_def_replace(&intr->def, nir_imm_vec2(&b, 0.25, 0.75));
               }
            }
         }
      }
      nir_opt_constant_folding(b.shader);
      nir_opt_dce(b.shader);
      unsigned depth_stores = 0, memory_stores = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            auto *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_ssbo) {
               ++memory_stores;
               EXPECT_EQ(nir_src_as_uint(intr->src[0]), 0xf00du);
            } else if (intr->intrinsic == nir_intrinsic_store_output) {
               ASSERT_EQ(nir_intrinsic_io_semantics(intr).location,
                         FRAG_RESULT_DEPTH);
               ++depth_stores;
               EXPECT_EQ(memory_stores, memory ? 1u : 0u);
               ASSERT_TRUE(nir_src_is_const(intr->src[0]));
               EXPECT_FLOAT_EQ(nir_src_as_float(intr->src[0]),
                               never ? 2.0f : explicit_depth ? 0.75f : 0.5f);
            }
         }
      }
      EXPECT_EQ(depth_stores,
                explicit_depth || (!early && (memory || feedback)) ? 1u : 0u);
      EXPECT_EQ(memory_stores, memory ? 1u : 0u);
      ralloc_free(b.shader);
   }
}

TEST_F(Apple9MSAA, TextureDescriptorPreservesSampleCountAndAddress)
{
   for (unsigned samples : {1u, 2u, 4u}) {
      struct AGX_TEXTURE cfg = {AGX_TEXTURE_header};
      cfg.dimension = samples == 1 ? AGX_TEXTURE_DIMENSION_2D
                                   : AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
      cfg.layout = AGX_LAYOUT_TWIDDLED;
      cfg.width = 33;
      cfg.height = 17;
      cfg.depth = 1;
      cfg.address = UINT64_C(0x1000024000);
      cfg.samples = samples == 4 ? AGX_SAMPLE_COUNT_4 : AGX_SAMPLE_COUNT_2;
      struct agx_texture_packed input;
      AGX_TEXTURE_pack(reinterpret_cast<uint32_t *>(&input), &cfg);
      uint8_t output[32];
      ASSERT_TRUE(agx_apple9_translate_texture(output, &input));
      uint32_t header;
      uint64_t address;
      memcpy(&header, output + 4, 4);
      memcpy(&address, output + 8, 8);
      EXPECT_EQ(!!(header & (1u << 24)), samples == 4);
      EXPECT_EQ(address, cfg.address >> 4);
      EXPECT_EQ(output[0] & 15, cfg.dimension);
   }
}

TEST_F(Apple9MSAA, RGBA8FourSampleGeometryMatchesNativeKernelContract)
{
   const pipe_format format = PIPE_FORMAT_R8G8B8A8_UNORM;
   auto tib = agx_apple9_build_tilebuffer_layout(&format, 1, 4, false);
   drm_asahi_cmd_render cmd = {};
   agx_tilebuffer_set_drm_cmd(&cmd, &tib);
   EXPECT_EQ(tib.sample_size_B, 8);
   EXPECT_EQ(tib.nr_samples, 4);
   EXPECT_EQ(hk_apple9_test_sample_positions(4), UINT64_C(0xeaa26e26));
   EXPECT_EQ(cmd.utile_width_px, 32);
   EXPECT_EQ(cmd.utile_height_px, 16);
   EXPECT_EQ(agx_tilebuffer_total_size(&tib), 16384u);
   unsigned utile_config = (cmd.utile_width_px / 16) << 12 |
                           (cmd.utile_height_px / 16) << 14 |
                           util_logbase2(tib.nr_samples);
   unsigned blocks = DIV_ROUND_UP(agx_tilebuffer_total_size(&tib), 2048);
   EXPECT_EQ(utile_config, 0x6002u);
   EXPECT_EQ(blocks, 8u);
   EXPECT_EQ((tib.nr_samples << 16) | blocks, 0x40008u);
}

TEST_F(Apple9MSAA, OneAndTwoSamplesKeepNative32PixelTileHeight)
{
   for (unsigned samples : {1u, 2u}) {
      const pipe_format format = PIPE_FORMAT_R8G8B8A8_UNORM;
      auto tib = agx_apple9_build_tilebuffer_layout(&format, 1, samples, false);
      drm_asahi_cmd_render cmd = {};
      agx_tilebuffer_set_drm_cmd(&cmd, &tib);
      EXPECT_EQ(tib.nr_samples, samples);
      EXPECT_EQ(tib.sample_size_B, 8);
      EXPECT_EQ(cmd.utile_width_px, 32);
      EXPECT_EQ(cmd.utile_height_px, 32);
      EXPECT_EQ(hk_apple9_test_sample_positions(samples),
                samples == 1 ? 0x88u : 0x44ccu);
   }
}

TEST_F(Apple9MSAA, NativeShapesAdmitWideMSAAWithoutChangingSingleSample)
{
   const pipe_format formats[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_B8G8R8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_SRGB, PIPE_FORMAT_B8G8R8A8_SRGB,
      PIPE_FORMAT_R10G10B10A2_UNORM, PIPE_FORMAT_B10G10R10A2_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R11G11B10_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_SINT,
   };
   for (auto format : formats)
   for (unsigned samples : {1u, 2u, 4u}) {
      SCOPED_TRACE(format);
      SCOPED_TRACE(samples);
      auto tib = agx_apple9_build_tilebuffer_layout(&format, 1, samples, false);
      bool wide = format == PIPE_FORMAT_R32G32B32A32_FLOAT ||
                  format == PIPE_FORMAT_R32G32B32A32_UINT ||
                  format == PIPE_FORMAT_R32G32B32A32_SINT;
      EXPECT_FALSE(tib.spilled[0]);
      EXPECT_EQ(tib.sample_size_B, wide ? 16 : 8);
      EXPECT_EQ(tib.tile_size, samples == 4 ? 32 * 16 : 32 * 32);
      EXPECT_LE(agx_tilebuffer_total_size(&tib), 32768u);
      if (samples == 1) {
         auto original = agx_build_tilebuffer_layout(&format, 1, 1, false);
         EXPECT_EQ(tib.sample_size_B, original.sample_size_B);
         EXPECT_EQ(tib.tile_size, original.tile_size);
      }
   }
}

TEST_F(Apple9MSAA, BlendSpecializationsDistinguishSampleCounts)
{
   vk_color_blend_state cb = {};
   cb.attachment_count = 1;
   cb.color_write_enables = 1;
   cb.attachments[0].write_mask = 15;
   agx_apple9_blend previous = {};
   for (unsigned samples : {1u, 2u, 4u}) {
      agx_apple9_blend blend = {};
      hk_apple9_test_blend_state(&cb, VK_FORMAT_R8G8B8A8_UNORM,
                                       samples, &blend);
      EXPECT_EQ(blend.samples, samples);
      EXPECT_NE(memcmp(&previous, &blend, sizeof(blend)), 0);
      previous = blend;
   }
}

TEST_F(Apple9MSAA, AbsentColorAttachmentDiscardsDeclaredColor)
{
   for (unsigned count : {0u, 1u})
   for (unsigned samples : {1u, 2u, 4u}) {
      vk_color_blend_state cb = {};
      cb.attachment_count = count;
      cb.color_write_enables = 1;
      cb.attachments[0].write_mask = 15;
      agx_apple9_blend blend = {};
      hk_apple9_test_blend_state(&cb, VK_FORMAT_UNDEFINED,
                                       samples, &blend);
      EXPECT_EQ(blend.format, PIPE_FORMAT_NONE);
      EXPECT_TRUE(blend.color_attachment_unused);
      EXPECT_EQ(blend.samples, samples);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "unused color attachment");
      nir_store_output(&b, nir_imm_vec4(&b, 1, 0, 0, 1), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_shader_part part = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment_blend(
         b.shader, nullptr, &blend, &part, &reason)) << (reason ? reason : "");
      unsigned stores = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block)
            stores += instr->type == nir_instr_type_intrinsic &&
               nir_instr_as_intrinsic(instr)->intrinsic ==
                  nir_intrinsic_store_local_pixel_agx;
      }
      EXPECT_EQ(stores, 0u);
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST_F(Apple9MSAA, ActualRuntimeResolveVertexCompiles)
{
   nir_shader *nir = vk_meta_draw_rects_vs_nir(nullptr, false);
   nir->options = &agx_nir_options;
   hk_apple9_test_preprocess_nir(nir);
   agx_apple9_vertex_layout layout = {};
   layout.format[0] = PIPE_FORMAT_R32G32B32A32_UINT;
   layout.stride[0] = 16;
   layout.draw_params_ubo = AGX_APPLE9_GRAPHICS_SYSVAL_BINDING;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_vertex_inputs(nir, &layout, &compiled, &reason))
      << (reason ? reason : "");
   EXPECT_TRUE(compiled.info.writes_layer_viewport);
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST_F(Apple9MSAA, TileHelpersCompileForEveryAdmittedColorFormat)
{
   const pipe_format formats[] = {
      PIPE_FORMAT_NONE,
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R10G10B10A2_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R11G11B10_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_SINT,
   };
   const VkClearColorValue clear = {{1.5f, 0.328125f, 19.5f, 0.0f}};
   for (auto format : formats)
   for (unsigned samples : {1u, 2u, 4u})
   for (unsigned op : {0u, 1u, 2u, 3u}) {
      SCOPED_TRACE(format);
      SCOPED_TRACE(samples);
      SCOPED_TRACE(op);
      agx_apple9_tilebuffer_key key = {};
      key.formats[0] = format;
      key.samples = samples;
      key.count = 1;
      key.mask = format == PIPE_FORMAT_NONE ? 0 : 1;
      key.store = op == 2;
      key.clear_mask = op == 0 ? 1 : 0;
      nir_shader *nir;
      if (op == 3 && format != PIPE_FORMAT_NONE) {
         /* Rectangles/scissors now select invocations in the runtime's draw
          * path. Its clear fragment shader does not reload outside the rect. */
         nir = hk_apple9_test_clear_shader(vk_format_from_pipe_format(format));
         nir->options = &agx_nir_options;
         hk_apple9_test_preprocess_nir(nir);
         nir_lower_explicit_io(nir, nir_var_mem_push_const,
                              nir_address_format_32bit_offset);
      } else {
         nir = agx_nir_apple9_tilebuffer(&agx_nir_options, &key);
      }
      /* Supply the producer's clear/push values and zero mip/layer inputs.
       * All format packing and sample addressing remain production NIR. */
      nir_opt_constant_folding(nir);
      nir_builder b = nir_builder_create(nir_shader_get_entrypoint(nir));
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            auto *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_ubo &&
                intr->intrinsic != nir_intrinsic_load_push_constant)
               continue;
            ASSERT_TRUE(nir_src_is_const(intr->src[0]));
            if (intr->intrinsic == nir_intrinsic_load_ubo) {
               ASSERT_TRUE(nir_src_is_const(intr->src[1]));
               ASSERT_EQ(nir_src_as_uint(intr->src[0]), 0u);
               ASSERT_EQ(nir_src_as_uint(intr->src[1]),
                         intr->def.num_components == 4 ? 64u : 0u);
            } else {
               ASSERT_EQ(nir_intrinsic_base(intr), 0u);
               ASSERT_EQ(nir_src_as_uint(intr->src[0]), 0u);
            }
            b.cursor = nir_before_instr(instr);
            nir_def *value;
            if (intr->def.num_components == 4) {
               value = nir_imm_ivec4(&b, clear.int32[0], clear.int32[1],
                                    clear.int32[2], clear.int32[3]);
            } else {
               ASSERT_EQ(intr->def.num_components, 2);
               value = nir_imm_ivec2(&b, 0, 0);
            }
            nir_def_replace(&intr->def, value);
         }
      }
      nir_opt_constant_folding(nir);
      nir_opt_dce(nir);
      /* Verify the NIR sample addressing before native compilation mutates it. */
      unsigned stores = 0, masks = 0, fetches = 0, exports = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               auto *tex = nir_instr_as_tex(instr);
               EXPECT_EQ(tex->op, samples > 1 ? nir_texop_txf_ms : nir_texop_txf);
               ++fetches;
            } else if (instr->type == nir_instr_type_intrinsic) {
               auto *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
                  ++stores;
                  masks |= nir_src_as_uint(intr->src[1]);
               } else if (intr->intrinsic == nir_intrinsic_image_store_block_agx) {
                  EXPECT_EQ(nir_intrinsic_image_dim(intr),
                            samples > 1 ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D);
                  ++exports;
               }
            }
         }
      }
      EXPECT_EQ(exports, format != PIPE_FORMAT_NONE && op == 2 ? 1u : 0u);
      EXPECT_EQ(fetches, format != PIPE_FORMAT_NONE && op == 1 ? samples : 0u);
      EXPECT_EQ(stores, format == PIPE_FORMAT_NONE || op >= 2 ? 0u : samples);
      if (stores) {
         EXPECT_EQ(masks, BITFIELD_MASK(samples));
      }
      agx_apple9_blend blend = {};
      vk_color_blend_state cb = {};
      cb.attachment_count = 1;
      cb.color_write_enables = 1;
      cb.attachments[0].write_mask = op == 3 ? 15 : 0;
      hk_apple9_test_blend_state(
         &cb, vk_format_from_pipe_format(format), samples, &blend);
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment_blend(
         nir, nullptr, &blend, &compiled, &reason)) << (reason ? reason : "");
      if (format == PIPE_FORMAT_R11G11B10_FLOAT && (op == 0 || op == 3)) {
         nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               auto *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_store_local_pixel_agx)
                  continue;
               ASSERT_NO_FATAL_FAILURE(expect_packed_float(
                  intr->src[0], 0x3e0u | (0x354u << 11) | (0x267u << 22)));
            }
         }
      }
      EXPECT_EQ(compiled.info.apple9_resource_count,
                format != PIPE_FORMAT_NONE && op == 1 ? 1u : 0u);
      EXPECT_EQ(compiled.info.apple9_image_mask,
                format != PIPE_FORMAT_NONE && op == 2 ? 1u : 0u);
      EXPECT_GT(compiled.info.binary_size, 0u);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST_F(Apple9MSAA, ActualRuntimeColorResolveCompilesWithNativeDescriptors)
{
   const VkFormat formats[] = {
      VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM,
      VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB,
      VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32,
      VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8_UNORM,
      VK_FORMAT_B10G11R11_UFLOAT_PACK32,
      VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT,
      VK_FORMAT_R32G32B32A32_SINT,
   };
   for (auto format : formats)
   for (auto samples : {VK_SAMPLE_COUNT_2_BIT, VK_SAMPLE_COUNT_4_BIT}) {
      SCOPED_TRACE(format);
      SCOPED_TRACE(samples);
      vk_meta_blit_key key = {};
      key.dim = GLSL_SAMPLER_DIM_MS;
      key.src_samples = samples;
      bool integer = format == VK_FORMAT_R32G32B32A32_UINT ||
                     format == VK_FORMAT_R32G32B32A32_SINT;
      key.resolve_mode = integer ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT
                                 : VK_RESOLVE_MODE_AVERAGE_BIT;
      key.dst_format = format;
      key.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
      nir_shader *nir = vk_meta_build_blit_shader(&key);
      nir->options = &agx_nir_options;
      hk_apple9_test_preprocess_nir(nir);
      nir_lower_explicit_io(nir, nir_var_mem_push_const,
                           nir_address_format_32bit_offset);
      auto *layout = static_cast<hk_descriptor_set_layout *>(calloc(
         1, sizeof(hk_descriptor_set_layout) +
            2 * sizeof(hk_descriptor_set_binding_layout)));
      layout->binding_count = 2;
      layout->binding[0].type = VK_DESCRIPTOR_TYPE_SAMPLER;
      layout->binding[0].array_size = 1;
      layout->binding[0].stride = 128;
      layout->binding[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      layout->binding[1].array_size = 1;
      layout->binding[1].stride = 128;
      vk_descriptor_set_layout *sets[] = {&layout->vk};
      const char *reason = nullptr;
      hk_apple9_test_lower_nir(nir, &vk_robustness_disabled, 1, sets);
      ASSERT_TRUE(hk_apple9_test_prepare_native_nir(nir));
      nir_opt_constant_folding(nir);
      nir_opt_copy_prop(nir);
      nir_opt_dce(nir);
      agx_apple9_blend blend = {};
      vk_color_blend_state cb = {};
      cb.attachment_count = 1;
      cb.color_write_enables = 1;
      cb.attachments[0].write_mask = 15;
      hk_apple9_test_blend_state(&cb, format, 1, &blend);
      agx_shader_part compiled = {};
      ASSERT_TRUE(agx_compile_apple9_fragment_blend(
         nir, nullptr, &blend, &compiled, &reason)) << (reason ? reason : "");
      /* Native descriptor handles are loaded through UBO 0, without a
       * fixed texture-table slot. */
      EXPECT_EQ(compiled.info.apple9_texture_mask, 0u);
      bool root = false;
      for (unsigned i = 0; i < compiled.info.apple9_resource_count; ++i)
         root |= compiled.info.apple9_resource_binding[i] == 0;
      EXPECT_TRUE(root);
      EXPECT_EQ(compiled.info.apple9_resource_ssbo_mask, 0u);
      EXPECT_TRUE(compiled.info.apple9_uses_texel_fetch);
      EXPECT_GT(compiled.info.binary_size, 0u);
      free(compiled.binary);
      free(layout);
      ralloc_free(nir);
   }
}

TEST_F(Apple9MSAA, PackedFloatAttachmentPreservesRangeAndOpaqueDestinationAlpha)
{
   /* These exactly representable UFLOAT values have independent bit oracles:
    * five exponent bits (bias 15), six R/G or five B fraction bits. Alpha is
    * absent, so DST_ALPHA must multiply RGB by one even when source alpha is 0.
    */
   const struct {
      float color[4];
      uint32_t expected;
   } cases[] = {
      {{1.5f, 0.328125f, 19.5f, 0.0f},
       (15u * 64 + 32) | ((13u * 64 + 20) << 11) | ((19u * 32 + 7) << 22)},
      {{65024.0f, 0x1p-20f, 64512.0f, 0.0f},
       (30u * 64 + 63) | (1u << 11) | ((30u * 32 + 31) << 22)},
   };
   for (const auto &test : cases)
   for (unsigned samples : {1u, 2u, 4u})
   for (bool destination_alpha : {false, true}) {
      vk_color_blend_state cb = {};
      cb.attachment_count = 1;
      cb.color_write_enables = 1;
      auto &rt = cb.attachments[0];
      rt.write_mask = 15;
      rt.blend_enable = destination_alpha;
      rt.src_color_blend_factor = VK_BLEND_FACTOR_DST_ALPHA;
      rt.dst_color_blend_factor = VK_BLEND_FACTOR_ZERO;
      rt.src_alpha_blend_factor = VK_BLEND_FACTOR_ONE;
      rt.dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO;
      agx_apple9_blend blend = {};
      hk_apple9_test_blend_state(&cb, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                                       samples, &blend);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "packed float attachment");
      nir_store_output(&b, nir_imm_vec4(&b, test.color[0], test.color[1],
                                        test.color[2], test.color[3]),
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment_blend(
         b.shader, nullptr, &blend, &compiled, &reason)) << (reason ? reason : "");
      unsigned stores = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            auto *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_local_pixel_agx)
               continue;
            ++stores;
            ASSERT_NO_FATAL_FAILURE(
               expect_packed_float(intr->src[0], test.expected));
            EXPECT_EQ(nir_intrinsic_format(intr), PIPE_FORMAT_R32_UINT);
         }
      }
      EXPECT_EQ(stores, 1u);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST_F(Apple9MSAA, PackedFloatBlendAndMaskCompileWithDestinationReads)
{
   for (unsigned samples : {1u, 2u, 4u})
   for (unsigned mask : {5u, 15u}) {
      vk_color_blend_state cb = {};
      cb.attachment_count = 1;
      cb.color_write_enables = 1;
      auto &rt = cb.attachments[0];
      rt.write_mask = mask;
      rt.blend_enable = true;
      rt.src_color_blend_factor = rt.dst_color_blend_factor = VK_BLEND_FACTOR_ONE;
      rt.src_alpha_blend_factor = rt.dst_alpha_blend_factor = VK_BLEND_FACTOR_ONE;
      agx_apple9_blend blend = {};
      hk_apple9_test_blend_state(&cb, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                                       samples, &blend);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "packed float blend");
      nir_store_output(&b, nir_load_frag_coord(&b), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment_blend(
         b.shader, nullptr, &blend, &compiled, &reason)) << (reason ? reason : "");
      EXPECT_TRUE(compiled.info.apple9_reads_tile);
      EXPECT_GT(compiled.info.binary_size, 0u);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}
