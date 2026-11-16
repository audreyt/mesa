/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_draw.c"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); failures++; } } while (0)
static unsigned failures, copies, resets;

void __wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                               enum agx_barrier barrier,
                               enum libagx_program idx, void *data, size_t size);
void
__wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                          enum agx_barrier barrier,
                          enum libagx_program idx, void *data, size_t size)
{
   CHECK(barrier & AGX_PREGFX);
   if (idx == LIBAGX_HEAP_RESET)
      resets++;
   else
      copies++;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   if (!dev || !cmd || !pool)
      return 1;
   const uint64_t base = 0x1000000000ull, end = base + (1ull << 32);
   struct agx_va va = {.addr = base + 0x20000000};
   struct agx_bo bo = {.va = &va, .size = 128 * 1024 * 1024};
   struct poly_heap heap = {.base = va.addr, .size = bo.size};
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = base;
   dev->heap = &bo;
   dev->heap_va = &va;
   va.size_B = bo.size;
   dev->heap_init_once.called = true;
   dev->rodata.heap = (uintptr_t)&heap;
   cmd->vk.base.device = &dev->vk;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = (uintptr_t)pool;
   list_inithead(&cmd->control_streams);

   const enum agx_index_size sizes[] = {
      AGX_INDEX_SIZE_U8, AGX_INDEX_SIZE_U16, AGX_INDEX_SIZE_U32,
   };
   const struct {
      enum agx_index_size size;
      uint32_t first;
      uint64_t offset;
      uint32_t remaining;
   } ranges[] = {
      {AGX_INDEX_SIZE_U8, UINT32_MAX, UINT64_C(0xffffffff), 0},
      {AGX_INDEX_SIZE_U16, 0x80000000u, UINT64_C(0x100000000), 0},
      {AGX_INDEX_SIZE_U32, 0x40000000u, UINT64_C(0x100000000), 0},
      {AGX_INDEX_SIZE_U32, UINT32_MAX, UINT64_C(0x3fffffffc), 0},
      {AGX_INDEX_SIZE_U32, 15, 60, 4},
      {AGX_INDEX_SIZE_U32, 16, 64, 0},
      {AGX_INDEX_SIZE_U32, 17, 68, 0},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(ranges); ++i) {
      struct agx_draw indexed = agx_draw_indexed(
         1, 1, ranges[i].first, 0, 0, base, 64, ranges[i].size, false);
      CHECK(agx_indices_to_B(indexed.start, indexed.index_size) ==
            ranges[i].offset);
      CHECK(agx_draw_index_buffer(indexed) == base + ranges[i].offset);
      CHECK(agx_draw_index_range_B(indexed) == ranges[i].remaining);
      indexed.b = agx_grid_indirect(base + 128);
      CHECK(agx_draw_index_buffer(indexed) == base);
      CHECK(agx_draw_index_range_B(indexed) == 64);
   }
   dev->vk.enabled_features.robustBufferAccess2 = true;
   for (unsigned i = 0; i < ARRAY_SIZE(sizes); i++) {
      unsigned size = agx_index_size_to_B(sizes[i]);
      struct agx_draw fit = agx_draw_indexed(
         18, 1, 3, 0, 0, base + 64, 21 * size, sizes[i], false);
      CHECK(!hk_apple9_needs_index_copy(cmd, fit));
      fit.index_buffer_range_B--;
      CHECK(hk_apple9_needs_index_copy(cmd, fit) == (size != 1));
      fit.index_buffer_range_B = 21 * size - (size == 1 ? 2 : 1);
      CHECK(hk_apple9_needs_index_copy(cmd, fit));
      fit.index_buffer = 0x10000;
      CHECK(hk_apple9_needs_index_copy(cmd, fit));
   }
   struct agx_draw overflow = agx_draw_indexed(
      UINT32_MAX, 1, UINT32_MAX, 0, 0, base, UINT32_MAX,
      AGX_INDEX_SIZE_U32, false);
   CHECK(hk_apple9_needs_index_copy(cmd, overflow));
   overflow.start = 3;
   overflow.index_buffer_range_B = 63;
   CHECK(hk_apple9_needs_index_copy(cmd, overflow));
   overflow.start = UINT32_MAX;
   overflow.b.count[0] = 1;
   overflow.index_buffer_range_B = 16;
   CHECK(hk_apple9_needs_index_copy(cmd, overflow));
   overflow.index_buffer = UINT64_MAX - 1;
   CHECK(hk_apple9_needs_index_copy(cmd, overflow));
   overflow.index_buffer = end - 4;
   overflow.index_buffer_range_B = 8;
   overflow.start = 0;
   overflow.b.count[0] = 2;
   CHECK(hk_apple9_needs_index_copy(cmd, overflow));
   overflow.index_buffer = base;
   overflow.index_buffer_range_B = 32;
   overflow.b.count[0] = 0;
   CHECK(!hk_apple9_needs_index_copy(cmd, overflow));
   dev->vk.enabled_features.robustBufferAccess2 = false;
   puts("firstIndex, short ranges and arithmetic limits use the live copy policy");

   struct agx_draw draw = agx_draw_indexed(3, 1, 0, 0, 0, end + 0x4000, 64,
                                           AGX_INDEX_SIZE_U32, false);
   CHECK(hk_apple9_needs_index_copy(cmd, draw));
   struct agx_draw copied = hk_apple9_robust_index(cmd, draw);
   CHECK(copies == 1 && copied.indexed && agx_is_indirect(copied.b));
   CHECK(resets == 1);
   CHECK(copied.index_buffer == va.addr && copied.index_buffer_range_B == bo.size);
   CHECK(copied.index_buffer >= base && copied.index_buffer < end);
   puts("high-VA binding: existing robust copy returns a reachable heap draw");

   draw.index_buffer = base - 64;
   CHECK(hk_apple9_needs_index_copy(cmd, draw));
   draw.index_buffer = end - 16;
   draw.start = 8;
   CHECK(hk_apple9_needs_index_copy(cmd, draw));
   puts("binding starts in aperture, firstIndex crosses its end: must copy");
   draw.start = 0;
   CHECK(hk_apple9_needs_index_copy(cmd, draw));
   draw.b = agx_grid_indirect(0x40000000);
   CHECK(hk_apple9_needs_index_copy(cmd, draw));
   draw.index_buffer = base + 0x4000;
   CHECK(!hk_apple9_needs_index_copy(cmd, draw));
   draw.index_buffer_range_B = 0;
   CHECK(!hk_apple9_needs_index_copy(cmd, draw));

   if (cmd->current_cs.post_gfx) {
      util_dynarray_fini(&cmd->current_cs.post_gfx->imm_writes);
      free(cmd->current_cs.post_gfx);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   free(pool);
   free(cmd);
   free(dev);
   return failures != 0;
}
