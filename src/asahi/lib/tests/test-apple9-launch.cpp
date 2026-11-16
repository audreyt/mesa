/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <array>
#include <cstring>
#include "gtest/gtest.h"

extern "C" {
#include "agx_apple9_launch.h"
#include "asahi/compiler/agx_apple9_profile.h"
}

static agx_apple9_launch_parameters
parameters()
{
   agx_apple9_launch_parameters p = {};
   p.shader_base = 0x10000000000ull;
   p.resource_table = p.shader_base + 0x208000;
   p.entry_offset = 0x12340;
   p.publication_count = 9;
   p.frame_extent_a = 4;
   p.frame_extent_b = 4096;
   p.tile_bytes = 24;
   p.samples = 1;
   p.resource_count = 8;
   /* Compute launches materialize their root words. */
   static uint32_t roots[2 * (AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE +
                              AGX_APPLE9_COMPUTE_DIRECT_MAX_RESOURCES)];
   p.root_words = roots;
   return p;
}

static unsigned
read16(const uint8_t *p)
{
   return p[0] | (p[1] << 8);
}

TEST(Apple9Launcher, AllStagesUseLiveEntryAndIndependentFrameExtents)
{
   auto p = parameters();
   for (auto stage : {AGX_APPLE9_LAUNCH_VERTEX, AGX_APPLE9_LAUNCH_FRAGMENT,
                      AGX_APPLE9_LAUNCH_COMPUTE}) {
      std::array<uint8_t, 1024> out;
      out.fill(0xa5);
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(), stage, &p));
      unsigned call = agx_apple9_launch_call_offset(stage, p.resource_count);
      unsigned entry = out[call] | (out[call + 1] << 8) | (out[call + 2] << 16);
      EXPECT_EQ(entry, 0x246aau); /* Entry above the former 64 KiB boundary. */
      EXPECT_EQ(read16(out.data() + call + 11), 0x280u);
      EXPECT_EQ(read16(out.data() + call + 13), 4u);
      EXPECT_EQ(read16(out.data() + call + 15), 4096u);
      EXPECT_EQ(out[call + 17], 0);
      EXPECT_EQ(out.back(), 0);
   }
}

TEST(Apple9Launcher, EntryTargetsCoverCompactArena)
{
   auto p = parameters();
   for (auto stage : {AGX_APPLE9_LAUNCH_VERTEX, AGX_APPLE9_LAUNCH_FRAGMENT,
                      AGX_APPLE9_LAUNCH_COMPUTE}) {
      for (unsigned offset : {0x40000u, 0x700000u, 0x7f8000u, 0x7fffeau}) {
         p.entry_offset = offset;
         std::array<uint8_t, 1024> out;
         ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(), stage, &p));
         unsigned call = agx_apple9_launch_call_offset(stage, p.resource_count);
         unsigned target = out[call] | (out[call + 1] << 8) |
                           (out[call + 2] << 16);
         EXPECT_EQ(target, 2 * offset + 0x2a);
      }
   }
}

TEST(Apple9Launcher, ResourceCountsRespectAllocationBounds)
{
   auto p = parameters();
   std::array<uint8_t, 1056> guarded;
   for (unsigned resources = 1; resources <= 18; ++resources) {
      p.resource_count = resources;
      guarded.fill(0xa5);
      ASSERT_TRUE(agx_apple9_launch_build(guarded.data() + 16, 1024,
                                          AGX_APPLE9_LAUNCH_COMPUTE, &p));
      for (unsigned i = 0; i < 16; ++i) {
         EXPECT_EQ(guarded[i], 0xa5);
         EXPECT_EQ(guarded[1040 + i], 0xa5);
      }
      auto *out = guarded.data() + 16;
      unsigned call =
         agx_apple9_launch_call_offset(AGX_APPLE9_LAUNCH_COMPUTE, resources);
      EXPECT_EQ(read16(out + call + 13), p.frame_extent_a);
      EXPECT_EQ(read16(out + call + 15), p.frame_extent_b);
      /* Publish exactly the group-count pointer plus active resource
       * pointers, then STOP. No state load or four-word reservation follows. */
      const unsigned words = 2 * (resources + 1);
      const unsigned arguments = call + 18;
      EXPECT_EQ(agx_apple9_compute_root_words(resources), words);
      for (unsigned word = 0; word < words; ++word) {
         const uint8_t *transfer = out + arguments + 4 * word;
         EXPECT_EQ(transfer[0] & 0xf, 0xbu);
         EXPECT_EQ((transfer[0] >> 4) | ((transfer[2] >> 6) << 4), word);
         EXPECT_EQ(transfer[1] >> 1, 18 + word);
         /* Literal roots leave nothing pending on slot 2. */
         EXPECT_EQ(transfer[3] & 0x40, 0);
      }
      EXPECT_EQ(read16(out + arguments + 4 * words), 0xeu);
      for (unsigned i = arguments + 4 * words + 4; i < 1024; ++i)
         EXPECT_EQ(out[i], 0u);
   }
}

TEST(Apple9Launcher, ResourceTablesUseFullAddresses)
{
   auto p = parameters();
   /* Decode the two literal operands independently of their emitter. */
   auto literal = [](const uint8_t *bytes) -> uint32_t {
      return (bytes[1] & 0x7fu) | ((bytes[4] & 0x1eu) << 6) |
             ((bytes[5] & 0xcu) << 9) | (uint32_t(bytes[6]) << 13) |
             ((bytes[7] & 0xfu) << 21) | ((bytes[3] & 0xfeu) << 24);
   };
   for (auto stage : {AGX_APPLE9_LAUNCH_VERTEX, AGX_APPLE9_LAUNCH_FRAGMENT,
                     AGX_APPLE9_LAUNCH_COMPUTE}) {
      for (uint64_t address : {p.shader_base - 0x1000,
                               p.shader_base + 0x1fffffc0,
                               p.shader_base + 0x20000000,
                               p.shader_base + 0x80000000,
                               p.shader_base + 0xfffff000}) {
         p.resource_table = address;
         std::array<uint8_t, 1024> out;
         ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(), stage, &p));
         EXPECT_EQ(literal(out.data()), uint32_t(address));
         EXPECT_EQ(literal(out.data() + 8), uint32_t(address >> 32));
      }
   }
}

TEST(Apple9Launcher, InvalidParametersCannotPublishPartialCode)
{
   auto p = parameters();
   auto reject = [&](agx_apple9_launch_parameters bad,
                     agx_apple9_launch_stage stage = AGX_APPLE9_LAUNCH_COMPUTE,
                     size_t capacity = 1024) {
      std::array<uint8_t, 1024> out;
      out.fill(0xa5);
      const auto saved = out;
      EXPECT_FALSE(agx_apple9_launch_build(out.data(), capacity, stage, &bad));
      EXPECT_EQ(out, saved);
   };
   reject(p, AGX_APPLE9_LAUNCH_COMPUTE, 1023);
   reject(p, AGX_APPLE9_LAUNCH_VERTEX, 255);
   reject(p, static_cast<agx_apple9_launch_stage>(AGX_APPLE9_LAUNCH_MESH + 1));
   auto bad = p;
   bad.entry_offset++;
   reject(bad);
   bad.entry_offset = 0x7fffec;
   reject(bad);
   bad.entry_offset = UINT32_MAX;
   reject(bad);
   bad = p;
   bad.publication_count = 1023;
   reject(bad);
   bad = p;
   for (unsigned count : {0u, 19u, UINT32_MAX}) {
      bad.resource_count = count;
      reject(bad);
   }
   bad = p;
   for (unsigned bytes : {1u, 64u, 129u, 384u, 8193u,
                          2u * AGX_APPLE9_MAX_SHARED_BYTES, UINT32_MAX}) {
      bad.threadgroup_memory_bytes = bytes;
      reject(bad);
   }
   bad.threadgroup_memory_bytes = 128;
   reject(bad, AGX_APPLE9_LAUNCH_VERTEX);
   reject(bad, AGX_APPLE9_LAUNCH_FRAGMENT);
   bad = p;
   for (unsigned samples : {0u, 3u, 8u}) {
      bad.samples = samples;
      reject(bad, AGX_APPLE9_LAUNCH_FRAGMENT);
   }
   bad = p;
   for (unsigned bytes : {129u}) {
      bad.tile_bytes = bytes;
      reject(bad, AGX_APPLE9_LAUNCH_FRAGMENT);
   }
}

TEST(Apple9Launcher, PublicationCapacityCoversPairBoundaries)
{
   auto p = parameters();
   const unsigned counts[] = {0, 1, 2, 8, 9, 36, 1022};
   const unsigned words[] = {0, 0x80, 0x80, 0x200, 0x280, 0x900, 0xff80};
   std::array<uint8_t, 256> out;
   for (unsigned i = 0; i < 7; ++i) {
      p.publication_count = counts[i];
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                          AGX_APPLE9_LAUNCH_VERTEX, &p));
      EXPECT_EQ(read16(out.data() + 68 + 11), words[i]);
   }
}

TEST(Apple9Launcher, ThreadgroupAllocationDoesNotChangeEntryOrFrame)
{
   auto p = parameters();
   std::array<uint8_t, 1024> initial, out;
   ASSERT_TRUE(agx_apple9_launch_build(initial.data(), initial.size(),
                                       AGX_APPLE9_LAUNCH_COMPUTE, &p));
   unsigned call = agx_apple9_launch_call_offset(AGX_APPLE9_LAUNCH_COMPUTE,
                                                 p.resource_count);
   for (unsigned bytes : {128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                          AGX_APPLE9_MAX_SHARED_BYTES}) {
      p.threadgroup_memory_bytes = bytes;
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                          AGX_APPLE9_LAUNCH_COMPUTE, &p));
      EXPECT_EQ(read16(out.data() + call - 6), ((bytes << 2) | 0x80) & 0xffff);
      EXPECT_EQ(out[call - 4], bytes >> 14);
      EXPECT_EQ(out[call - 3], 0);
      /* Only the allocation operand changes. */
      out[call - 6] = initial[call - 6];
      out[call - 5] = initial[call - 5];
      out[call - 4] = initial[call - 4];
      EXPECT_EQ(out, initial);
   }
}

TEST(Apple9Launcher, LargerThreadgroupAllocationsFitTheAllocationWord)
{
   auto p = parameters();
   const unsigned sizes[] = {2048, 4096, 8192};
   const unsigned words[] = {0x2080, 0x4080, 0x8080};
   std::array<uint8_t, 1024> initial, out;
   ASSERT_TRUE(agx_apple9_launch_build(initial.data(), initial.size(),
                                     AGX_APPLE9_LAUNCH_COMPUTE, &p));
   unsigned call = agx_apple9_launch_call_offset(AGX_APPLE9_LAUNCH_COMPUTE,
                                                p.resource_count);
   for (unsigned i = 0; i < 3; ++i) {
      SCOPED_TRACE(sizes[i]);
      p.threadgroup_memory_bytes = sizes[i];
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                        AGX_APPLE9_LAUNCH_COMPUTE, &p));
      EXPECT_EQ(read16(out.data() + call - 6), words[i]);
      out[call - 6] = initial[call - 6];
      out[call - 5] = initial[call - 5];
      EXPECT_EQ(out, initial);
   }
}

TEST(Apple9Launcher, TileModesDoNotChangeEntryOrResources)
{
   auto p = parameters();
   std::array<uint8_t, 256> initial, out;
   ASSERT_TRUE(agx_apple9_launch_build(initial.data(), initial.size(),
                                       AGX_APPLE9_LAUNCH_FRAGMENT, &p));
   struct {
      unsigned samples, bytes, mode, low;
   } cases[] = {{1, 4, 0x4b, 0x43}, {1, 8, 0x4b, 0x43},
               {1, 16, 0x4b, 0x45}, {2, 24, 0x4b, 0x46}, {4, 64, 0x43, 16},
               {1, 128, 0x4b, 0x61}, {1, 0, 0x4b, 0x41},
               {2, 0, 0x4b, 0x40}, {4, 0, 0x43, 0}};
   for (auto c : cases) {
      p.samples = c.samples;
      p.tile_bytes = c.bytes;
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                          AGX_APPLE9_LAUNCH_FRAGMENT, &p));
      EXPECT_EQ(out[68 - 7], c.low);
      EXPECT_EQ(out[68 - 6], c.mode);
      memcpy(out.data() + 68 - 10, initial.data() + 68 - 10, 8);
      EXPECT_EQ(out, initial);
   }
}

/* Multisampled fragment launches without colour targets keep a nonempty
 * per-sample tile allocation, as agx_build_tilebuffer_layout does. */
TEST(Apple9Launcher, MultisampledZeroTargetLaunchesKeepTileAllocation)
{
   auto p = parameters();
   std::array<uint8_t, 256> out;
   /* agx_apple9_tile_sample_bytes(0, samples): 0 at 1x, 8 when multisampled. */
   struct {
      unsigned samples, bytes, low;
   } cases[] = {{1, 0, 0x41}, {2, 8, 0x42}, {4, 8, 2}};
   for (auto c : cases) {
      p.samples = c.samples;
      p.tile_bytes = c.bytes;
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                          AGX_APPLE9_LAUNCH_FRAGMENT, &p));
      EXPECT_EQ(out[68 - 7], c.low);
   }
}

TEST(Apple9Launcher, PreambleTransferFollowsRootArguments)
{
   auto p = parameters();
   p.launch_address = p.shader_base + 0x204000;
   for (auto stage : {AGX_APPLE9_LAUNCH_VERTEX, AGX_APPLE9_LAUNCH_FRAGMENT,
                      AGX_APPLE9_LAUNCH_COMPUTE}) {
      for (uint64_t offset : {0x10000ull, 0xf0000000ull}) {
         p.preamble_address = p.shader_base + offset;
         std::array<uint8_t, 1024> out;
         ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(), stage, &p));
         unsigned words = stage == AGX_APPLE9_LAUNCH_COMPUTE
            ? agx_apple9_compute_root_words(p.resource_count)
            : AGX_APPLE9_GRAPHICS_ROOT_WORDS;
         unsigned at = agx_apple9_launch_call_offset(stage, p.resource_count) + 18 + 4 * words;
         EXPECT_EQ(read16(out.data() + at), 0x000fu);
         int64_t displacement = 0;
         for (unsigned i = 0; i < 6; ++i)
            displacement |= (uint64_t)out[at + 3 + i] << (8 * i);
         if (displacement & (INT64_C(1) << 47))
            displacement |= -(INT64_C(1) << 48);
         EXPECT_EQ(p.launch_address + at + displacement, p.preamble_address);
         EXPECT_EQ(read16(out.data() + at + 10), 0xeu);
         EXPECT_EQ(out.back(), 0);
      }
   }
}

TEST(Apple9Launcher, PreambleBoundsRejectWithoutWriting)
{
   auto p = parameters();
   p.launch_address = p.shader_base + 0x204000;
   std::array<uint8_t, 1024> out;
   for (uint64_t address : {p.shader_base - 2, p.shader_base + 1,
                            p.shader_base + UINT64_C(0x100000000)}) {
      p.preamble_address = address;
      out.fill(0xa5);
      auto before = out;
      EXPECT_FALSE(agx_apple9_launch_build(out.data(), out.size(), AGX_APPLE9_LAUNCH_COMPUTE, &p));
      EXPECT_EQ(out, before);
   }
   p.preamble_address = p.shader_base + 0x800000;
   p.launch_address = p.shader_base - 2;
   out.fill(0xa5);
   auto before = out;
   EXPECT_FALSE(agx_apple9_launch_build(out.data(), out.size(), AGX_APPLE9_LAUNCH_COMPUTE, &p));
   EXPECT_EQ(out, before);
}

/* Graphics retains its established wire hashes. Compute now materializes
 * roots, so decode that protocol and keep the independent allocation bytes. */
static uint64_t
fnv1a(const uint8_t *bytes, size_t count)
{
   uint64_t hash = 0xcbf29ce484222325ull;
   for (size_t i = 0; i < count; ++i) {
      hash ^= bytes[i];
      hash *= 0x100000001b3ull;
   }
   return hash;
}

struct Golden {
   agx_apple9_launch_stage stage;
   unsigned samples, tile_bytes, threadgroup;
   uint8_t alloc[8];
   uint64_t hash;
};

static const Golden goldens[] = {
   {AGX_APPLE9_LAUNCH_VERTEX, 1, 24, 0, {0x77,0x00,0x2a,0x41,0,0,0,0}, 0x598081d0090e4fefull},
   {AGX_APPLE9_LAUNCH_FRAGMENT, 1, 24, 0, {0x77,0x00,0x2a,0x47,0x4b,0,0,0}, 0x28a72854c5d9c1faull},
   {AGX_APPLE9_LAUNCH_FRAGMENT, 2, 24, 0, {0x77,0x00,0x2a,0x46,0x4b,0,0,0}, 0x1ce71feccc82dc89ull},
   {AGX_APPLE9_LAUNCH_FRAGMENT, 4, 16, 0, {0x77,0x00,0xaa,0x04,0x43,0,0,0}, 0x9cf46aa6080effefull},
   {AGX_APPLE9_LAUNCH_COMPUTE, 1, 24, 0, {0x77,0x00,0x2a,0x41,0,0,0,0}, 0x2850ccf67788c2a9ull},
   {AGX_APPLE9_LAUNCH_COMPUTE, 1, 24, 128, {0x77,0x00,0x2a,0x41,0x80,0x02,0,0}, 0x16cc7d7858ba5de3ull},
   {AGX_APPLE9_LAUNCH_COMPUTE, 1, 24, 256, {0x77,0x00,0x2a,0x41,0x80,0x04,0,0}, 0x113e865f4d67221dull},
   {AGX_APPLE9_LAUNCH_COMPUTE, 1, 24, 512, {0x77,0x00,0x2a,0x41,0x80,0x08,0,0}, 0x53701eaaa210ff21ull},
   {AGX_APPLE9_LAUNCH_COMPUTE, 1, 24, 1024, {0x77,0x00,0x2a,0x41,0x80,0x10,0,0}, 0xb9e585df6df7ffb9ull},
   {AGX_APPLE9_LAUNCH_COMPUTE, 1, 24, 2048, {0x77,0x00,0x2a,0x41,0x80,0x20,0,0}, 0x0ecddc0478d65089ull},
   {AGX_APPLE9_LAUNCH_COMPUTE, 1, 24, 4096, {0x77,0x00,0x2a,0x41,0x80,0x40,0,0}, 0xb192abf38027f569ull},
   {AGX_APPLE9_LAUNCH_COMPUTE, 1, 24, 8192, {0x77,0x00,0x2a,0x41,0x80,0x80,0,0}, 0xae6bd399ee4b57a9ull},
};

static void
check_goldens()
{
   for (const Golden &g : goldens) {
      SCOPED_TRACE(g.threadgroup);
      auto p = parameters();
      p.samples = g.samples;
      p.tile_bytes = g.tile_bytes;
      p.threadgroup_memory_bytes = g.threadgroup;
      std::array<uint8_t, 1024> out;
      out.fill(0xa5);
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(), g.stage, &p));
      unsigned call = agx_apple9_launch_call_offset(g.stage, p.resource_count);
      EXPECT_EQ(0, memcmp(out.data() + call - 10, g.alloc, 8));
      if (g.stage != AGX_APPLE9_LAUNCH_COMPUTE) {
         EXPECT_EQ(fnv1a(out.data(), out.size()), g.hash);
      } else {
         EXPECT_EQ(call, 170u);
         for (unsigned i = 0; i < 18; ++i) {
            unsigned at = 16 + 8 * i;
            EXPECT_EQ(out[at] & 15, 12);
            EXPECT_EQ(out[at + 1], 0x80);
            EXPECT_EQ(out[at + 2] & 63, 2);
            EXPECT_EQ((out[at] >> 4) | ((out[at + 2] >> 6) << 4), 18 + i);
            for (unsigned byte = 3; byte < 8; ++byte)
               EXPECT_EQ(out[at + byte], 0);
         }
         EXPECT_EQ(read16(out.data() + 260), 0xeu);
         for (unsigned at = 264; at < out.size(); ++at)
            EXPECT_EQ(out[at], 0);
      }
   }
}

TEST(Apple9LauncherGolden, ProductionEncodingIsUnchangedByDefault)
{
   check_goldens();
}

TEST(Apple9LauncherGolden, LargeAllocationDoesNotChangeFollowingSmallLaunches)
{
   auto p = parameters();
   p.threadgroup_memory_bytes = 32768;
   std::array<uint8_t, 1024> out;
   ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                     AGX_APPLE9_LAUNCH_COMPUTE, &p));
   check_goldens();
}

TEST(Apple9LauncherGolden, NativeLimitRejectsOversizeWithoutWriting)
{
   auto p = parameters();
   for (unsigned bytes : {32769u, 65536u, UINT32_MAX}) {
      p.threadgroup_memory_bytes = bytes;
      std::array<uint8_t, 1024> out;
      out.fill(0xa5);
      auto before = out;
      EXPECT_FALSE(agx_apple9_launch_threadgroup_memory_supported(bytes));
      EXPECT_FALSE(agx_apple9_launch_build(out.data(), out.size(),
                                           AGX_APPLE9_LAUNCH_COMPUTE, &p));
      EXPECT_EQ(out, before);
   }
   p.threadgroup_memory_bytes = 32768;
   std::array<uint8_t, 1024> out;
   EXPECT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                     AGX_APPLE9_LAUNCH_COMPUTE, &p));
}

TEST(Apple9LauncherGolden, NativeWideAllocationUsesTheThirdByte)
{
   auto p = parameters();
   std::array<uint8_t, 1024> base, out;
   ASSERT_TRUE(agx_apple9_launch_build(base.data(), base.size(),
                                       AGX_APPLE9_LAUNCH_COMPUTE, &p));
   unsigned call = agx_apple9_launch_call_offset(AGX_APPLE9_LAUNCH_COMPUTE,
                                                 p.resource_count);
   const struct {
      unsigned bytes;
      uint8_t word[4];
   } cases[] = {
      {16384, {0x80, 0x00, 0x01, 0x00}},
      {32768, {0x80, 0x00, 0x02, 0x00}},
   };
   for (const auto &c : cases) {
      SCOPED_TRACE(c.bytes);
      p.threadgroup_memory_bytes = c.bytes;
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                          AGX_APPLE9_LAUNCH_COMPUTE, &p));
      EXPECT_EQ(0, memcmp(out.data() + call - 6, c.word, 4));
      /* Only allocation bytes 4..7 differ from the unallocated launch. */
      memcpy(out.data() + call - 6, base.data() + call - 6, 4);
      EXPECT_EQ(out, base);
   }
   /* Beyond the native ceiling and non powers of two stay rejected. */
   for (unsigned bytes : {65536u, 24576u}) {
      p.threadgroup_memory_bytes = bytes;
      EXPECT_FALSE(agx_apple9_launch_build(out.data(), out.size(),
                                           AGX_APPLE9_LAUNCH_COMPUTE, &p));
   }
}

TEST(Apple9LauncherGolden, NativeEntryPositionMatchesStageAndInlineRoots)
{
   auto p = parameters();
   const agx_apple9_launch_stage stages[] = {
      AGX_APPLE9_LAUNCH_VERTEX, AGX_APPLE9_LAUNCH_FRAGMENT,
      AGX_APPLE9_LAUNCH_COMPUTE};
   const unsigned positions[] = {68, 68, 170};
   for (unsigned i = 0; i < 3; ++i) {
      std::array<uint8_t, 1024> out;
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(), stages[i], &p));
      EXPECT_EQ(agx_apple9_launch_call_offset(stages[i], p.resource_count),
                positions[i]);
      unsigned at = positions[i];
      EXPECT_EQ(out[at] | (out[at + 1] << 8) | (out[at + 2] << 16), 0x246aau);
   }
}

TEST(Apple9LauncherGolden, TemplateKeepsItsAllocationValueSnapshot)
{
   auto p = parameters();
   p.threadgroup_memory_bytes = 16384;
   agx_apple9_launch_template tmpl = {};
   ASSERT_TRUE(agx_apple9_launch_template_init(
      &tmpl, 1024, AGX_APPLE9_LAUNCH_COMPUTE, &p));
   p.threadgroup_memory_bytes = 32768;
   std::array<uint8_t, 1024> out;
   ASSERT_TRUE(agx_apple9_launch_template_emit(
      out.data(), &tmpl, p.resource_table, p.entry_offset, p.launch_address));
   EXPECT_EQ(read16(out.data() + 164), 0x80u);
   EXPECT_EQ(out[166], 1);
   EXPECT_EQ(out[167], 0);
   ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                     AGX_APPLE9_LAUNCH_COMPUTE, &p));
   EXPECT_EQ(read16(out.data() + 164), 0x80u);
   EXPECT_EQ(out[166], 2);
}
