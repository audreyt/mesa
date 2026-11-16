/* SPDX-License-Identifier: MIT */
#ifndef HK_APPLE9_TEST_H
#define HK_APPLE9_TEST_H

#include "compiler/nir/nir.h"
#include "vulkan/vulkan_core.h"

struct vk_pipeline_robustness_state;
struct vk_descriptor_set_layout;
struct vk_color_blend_state;
struct agx_apple9_blend;

#ifdef __cplusplus
extern "C" {
#endif

bool hk_apple9_test_lower_descriptors(
   nir_shader *nir, const struct vk_pipeline_robustness_state *rs,
   unsigned count, struct vk_descriptor_set_layout *const *sets);
void hk_apple9_test_lower_nir(
   nir_shader *nir, const struct vk_pipeline_robustness_state *rs,
   unsigned count, struct vk_descriptor_set_layout *const *sets);
bool hk_apple9_test_prepare_native_nir(nir_shader *nir);
void hk_apple9_test_preprocess_nir(nir_shader *nir);
void hk_apple9_test_late_zs(nir_shader *nir, bool zs_self_dep);

struct hk_apple9_test_storage_layout {
   uint32_t stride;
   uint32_t texel_size_offset;
   uint32_t components_offset;
   uint32_t store_encode_offset;
};

struct hk_apple9_test_storage_layout hk_apple9_test_get_storage_layout(void);
void hk_apple9_test_blend_state(const struct vk_color_blend_state *cb,
                               VkFormat format, unsigned samples,
                               struct agx_apple9_blend *blend);
nir_shader *hk_apple9_test_clear_shader(VkFormat format);
uint64_t hk_apple9_test_sample_positions(unsigned samples);

#ifdef __cplusplus
}
#endif
#endif
