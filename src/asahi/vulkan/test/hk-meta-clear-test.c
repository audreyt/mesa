/* SPDX-License-Identifier: MIT
 *
 * The runtime library is linked whole. Keep this test translation unit's
 * included entrypoints distinct while exercising the real static NIR builder.
 */
#define vk_meta_rect_cmp_layer test_meta_rect_cmp_layer
#define vk_meta_clear_attachments test_meta_clear_attachments
#define vk_meta_clear_rendering test_meta_clear_rendering
#define vk_meta_clear_color_image test_meta_clear_color_image
#define vk_meta_clear_depth_stencil_image test_meta_clear_depth_stencil_image
#include "../../../vulkan/runtime/vk_meta_clear.c"
#include "hk-apple9-test.h"

nir_shader *
hk_apple9_test_clear_shader(VkFormat format)
{
   const struct vk_meta_clear_key key = {
      .key_type = VK_META_OBJECT_KEY_CLEAR,
      .render = {
         .color_attachment_count = 1,
         .color_attachment_formats = {format},
      },
      .color_attachments_cleared = 1,
   };
   return build_clear_shader(&key);
}
