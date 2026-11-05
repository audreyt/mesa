/* SPDX-License-Identifier: MIT */

#include "zink_context.h"
#include "zink_screen.h"
#include "util/ralloc.h"
#include "util/u_helpers.h"
#include "util/u_inlines.h"
#include "util/u_upload_mgr.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <tuple>
#include <vector>

namespace {

struct buffer {
   struct pipe_resource base;
   std::vector<uint8_t> data;
};

static struct pipe_resource *
create_buffer(struct pipe_screen *screen, const struct pipe_resource *templ)
{
   auto *buf = new buffer;
   buf->base = *templ;
   buf->base.screen = screen;
   pipe_reference_init(&buf->base.reference, 1);
   buf->data.resize(templ->width0, 0xcd);
   return &buf->base;
}

static void
destroy_buffer(struct pipe_screen *, struct pipe_resource *resource)
{
   delete reinterpret_cast<buffer *>(resource);
}

static void *
map_buffer(struct pipe_context *, struct pipe_resource *resource,
           unsigned level, unsigned usage, const struct pipe_box *box,
           struct pipe_transfer **transfer)
{
   auto *buf = reinterpret_cast<buffer *>(resource);
   auto *map = new pipe_transfer{};
   map->resource = resource;
   map->level = level;
   map->usage = static_cast<enum pipe_map_flags>(usage);
   map->box = *box;
   *transfer = map;
   return buf->data.data() + box->x;
}

static void
unmap_buffer(struct pipe_context *, struct pipe_transfer *transfer)
{
   delete transfer;
}

static void
flush_buffer(struct pipe_context *, struct pipe_transfer *,
             const struct pipe_box *)
{
}

/* Exercise the production Zink restart entry point without creating a Vulkan
 * device. The downstream callback consumes actual uploaded index bytes and
 * Gallium draw IDs, including every range in a multi-draw submission.
 */
class prim_restart : public testing::TestWithParam<std::tuple<bool, bool, unsigned>> {
protected:
   struct zink_context *ctx = nullptr;
   struct zink_screen *screen = nullptr;
   std::vector<unsigned> indices;
   std::vector<unsigned> draw_ids;
   std::vector<int> biases;

   void SetUp() override
   {
      ctx = rzalloc(nullptr, struct zink_context);
      screen = rzalloc(nullptr, struct zink_screen);
      ASSERT_NE(ctx, nullptr);
      ASSERT_NE(screen, nullptr);
      screen->info.driver_props.driverID = VK_DRIVER_ID_MESA_HONEYKRISP;
      screen->base.resource_create = create_buffer;
      screen->base.resource_destroy = destroy_buffer;
      ctx->base.screen = &screen->base;
      ctx->base.priv = this;
      ctx->base.resource_release = u_default_resource_release;
      ctx->base.buffer_map = map_buffer;
      ctx->base.buffer_unmap = unmap_buffer;
      ctx->base.transfer_flush_region = flush_buffer;
      ctx->base.stream_uploader = u_upload_create_default(&ctx->base);
      ASSERT_NE(ctx->base.stream_uploader, nullptr);
      zink_init_draw_functions(ctx, screen);
      ctx->base.draw_vbo = consume;
   }

   void TearDown() override
   {
      u_upload_destroy(ctx->base.stream_uploader);
      for (unsigned i = 0; i < ARRAY_SIZE(ctx->program_cache); i++)
         _mesa_hash_table_fini(&ctx->program_cache[i], nullptr);
      for (unsigned i = 0; i < ARRAY_SIZE(ctx->program_lock); i++)
         simple_mtx_destroy(&ctx->program_lock[i]);
      for (unsigned i = 0; i < ARRAY_SIZE(ctx->mesh_cache); i++)
         _mesa_hash_table_fini(&ctx->mesh_cache[i], nullptr);
      for (unsigned i = 0; i < ARRAY_SIZE(ctx->mesh_lock); i++)
         simple_mtx_destroy(&ctx->mesh_lock[i]);
      ralloc_free(ctx);
      ralloc_free(screen);
   }

   static void consume(struct pipe_context *pipe, const struct pipe_draw_info *info,
                       unsigned drawid_offset,
                       const struct pipe_draw_indirect_info *indirect,
                       const struct pipe_draw_start_count_bias *draws,
                       unsigned num_draws)
   {
      auto *test = static_cast<prim_restart *>(pipe->priv);
      ASSERT_FALSE(info->primitive_restart);
      ASSERT_EQ(indirect, nullptr);
      struct pipe_resource *uploaded = nullptr;
      struct pipe_resource *resource = info->index.resource;
      unsigned offset = 0;
      if (info->has_user_indices) {
         ASSERT_TRUE(util_upload_index_buffer(pipe, info, &draws[0],
                                             &uploaded, &offset, 4));
         resource = uploaded;
      }
      const auto *buf = reinterpret_cast<buffer *>(resource);
      for (unsigned i = 0; i < num_draws; i++) {
         for (unsigned j = 0; j < draws[i].count; j++) {
            unsigned byte = offset + (draws[i].start + j) * info->index_size;
            ASSERT_LE(byte + info->index_size, buf->data.size());
            unsigned index = 0;
            if (info->index_size == 2) {
               uint16_t value;
               memcpy(&value, buf->data.data() + byte, sizeof(value));
               index = value;
            } else {
               memcpy(&index, buf->data.data() + byte, sizeof(index));
            }
            test->indices.push_back(index);
            test->draw_ids.push_back(drawid_offset +
                                     (info->increment_draw_id ? i : 0));
            test->biases.push_back(draws[i].index_bias);
         }
      }
      pipe_resource_reference(&uploaded, nullptr);
   }
};

TEST_P(prim_restart, preserves_logical_draws_and_all_index_ranges)
{
   const auto [user_indices, increment_draw_id, index_size] = GetParam();
   const unsigned restart = index_size == 2 ? 0xffff : 0xffffffff;
   /* Nonzero starts, two segments per logical draw, and distinct base vertices. */
   const unsigned values[] = {
      91, 92, 1, 2, 3, restart, 4, 5, 6,
      7, 8, 9, restart, 10, 11, 12,
   };
   std::vector<uint8_t> data(ARRAY_SIZE(values) * index_size);
   for (unsigned i = 0; i < ARRAY_SIZE(values); i++) {
      if (index_size == 2) {
         uint16_t value = values[i];
         memcpy(data.data() + i * index_size, &value, sizeof(value));
      } else {
         memcpy(data.data() + i * index_size, &values[i], sizeof(values[i]));
      }
   }

   struct pipe_draw_info info = {};
   info.mode = MESA_PRIM_TRIANGLE_STRIP;
   info.index_size = index_size;
   info.primitive_restart = true;
   info.restart_index = restart;
   info.has_user_indices = user_indices;
   info.increment_draw_id = increment_draw_id;
   info.instance_count = 1;
   info.index_bias_varies = true;
   struct pipe_resource *resource = nullptr;
   if (user_indices) {
      info.index.user = data.data();
   } else {
      struct pipe_resource templ = {};
      templ.target = PIPE_BUFFER;
      templ.width0 = data.size();
      resource = create_buffer(&screen->base, &templ);
      reinterpret_cast<buffer *>(resource)->data = data;
      info.index.resource = resource;
   }
   const struct pipe_draw_start_count_bias draws[] = {{2, 7, 3}, {9, 7, -2}};

   ctx->draw_vbo[0](&ctx->base, &info, 7, nullptr, draws, ARRAY_SIZE(draws));

   EXPECT_EQ(indices, (std::vector<unsigned>{1, 2, 3, 4, 5, 6,
                                            7, 8, 9, 10, 11, 12}));
   std::vector<unsigned> expected_ids(12, 7);
   if (increment_draw_id)
      std::fill(expected_ids.begin() + 6, expected_ids.end(), 8);
   EXPECT_EQ(draw_ids, expected_ids);
   EXPECT_EQ(biases, (std::vector<int>{3, 3, 3, 3, 3, 3,
                                     -2, -2, -2, -2, -2, -2}));
   pipe_resource_reference(&resource, nullptr);
}

INSTANTIATE_TEST_SUITE_P(index_buffers, prim_restart,
                        testing::Combine(testing::Bool(), testing::Bool(),
                                         testing::Values(2u, 4u)));

} // namespace
