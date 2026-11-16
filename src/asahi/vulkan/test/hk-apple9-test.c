/* SPDX-License-Identifier: MIT
 *
 * Follow the native C tests' include-the-owner pattern. This translation unit
 * gives C++ tests language access to actual static shader helpers; it adds no
 * driver export and contains no replacement lowering implementation.
 */
#include "../hk_shader.c"
#include "hk-apple9-test.h"

bool
hk_apple9_test_lower_descriptors(
   nir_shader *nir, const struct vk_pipeline_robustness_state *rs,
   unsigned count, struct vk_descriptor_set_layout *const *sets)
{
   struct hk_device dev = {.dev.chip = AGX_CHIP_G16G};
   struct vk_device *saved[HK_MAX_SETS];
   for (unsigned i = 0; i < count; ++i) {
      if (sets[i]) {
         saved[i] = sets[i]->base.device;
         sets[i]->base.device = &dev.vk;
      }
   }
   bool progress = hk_nir_lower_descriptors(nir, rs, count, sets, true);
   for (unsigned i = 0; i < count; ++i) {
      if (sets[i])
         sets[i]->base.device = saved[i];
   }
   return progress;
}

void
hk_apple9_test_lower_nir(
   nir_shader *nir, const struct vk_pipeline_robustness_state *rs,
   unsigned count, struct vk_descriptor_set_layout *const *sets)
{
   struct hk_instance instance = {0};
   struct hk_physical_device physical = {
      .vk.instance = &instance.vk,
      .dev.chip = AGX_CHIP_G16G,
   };
   struct hk_device dev = {
      .dev.chip = AGX_CHIP_G16G,
      .vk.physical = &physical.vk,
   };
   struct vk_device *saved[HK_MAX_SETS];
   for (unsigned i = 0; i < count; ++i) {
      if (sets[i]) {
         saved[i] = sets[i]->base.device;
         sets[i]->base.device = &dev.vk;
      }
   }
   hk_lower_nir(&dev, nir, rs, false, count, sets, 0);
   for (unsigned i = 0; i < count; ++i) {
      if (sets[i])
         sets[i]->base.device = saved[i];
   }
}

bool
hk_apple9_test_prepare_native_nir(nir_shader *nir)
{
   const char *reason = NULL;
   if (!hk_apple9_texture_bindings(nir) ||
       !agx_nir_lower_apple9_runtime_sampler_state(nir, &reason))
      return false;
   nir_shader_intrinsics_pass(nir, hk_apple9_lower_roots,
                              nir_metadata_control_flow, NULL);
   return true;
}

void
hk_apple9_test_preprocess_nir(nir_shader *nir)
{
   struct hk_physical_device physical = {.dev.chip = AGX_CHIP_G16G};
   hk_preprocess_nir_internal(&physical.vk, nir);
}

void
hk_apple9_test_late_zs(nir_shader *nir, bool zs_self_dep)
{
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   hk_apple9_late_zs(nir, zs_self_dep);
   nir_shader_intrinsics_pass(nir, hk_apple9_clamp_depth_export,
                              nir_metadata_control_flow, NULL);
}

struct hk_apple9_test_storage_layout
hk_apple9_test_get_storage_layout(void)
{
   return (struct hk_apple9_test_storage_layout){
      .stride = sizeof(struct hk_apple9_storage_descriptor),
      .texel_size_offset =
         offsetof(struct hk_apple9_storage_descriptor, metadata.size_B),
      .components_offset =
         offsetof(struct hk_apple9_storage_descriptor, metadata.component),
      .store_encode_offset =
         offsetof(struct hk_apple9_storage_descriptor, store_encode),
   };
}

void
hk_apple9_test_blend_state(const struct vk_color_blend_state *cb,
                          VkFormat format, unsigned samples,
                          struct agx_apple9_blend *blend)
{
   struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {format},
   };
   struct vk_multisample_state ms = {
      .rasterization_samples = samples,
      .sample_mask = ~0u,
   };
   struct vk_rasterization_state rs = {0};
   struct vk_graphics_pipeline_state state = {
      .rp = &rp, .ms = &ms, .cb = cb, .rs = &rs,
   };
   struct hk_apple9_graphics_key key;
   hk_apple9_graphics_key(&key, &state);
   *blend = key.blend[0];
}

uint64_t
hk_apple9_test_sample_positions(unsigned samples)
{
   return hk_default_sample_positions(samples);
}
