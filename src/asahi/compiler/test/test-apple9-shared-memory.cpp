/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_machine.h"
#include "agx_apple9_profile.h"
#include "agx_compile_apple9.h"

#include "asahi/lib/agx_apple9_launch.h"
#include "compiler/nir/nir_builder.h"
#include <gtest/gtest.h>

static bool
compile_shared(unsigned shared_size, agx_apple9_compute_profile *profile,
               const char **reason)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "native shared memory");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = shared_size;
   nir_def *lane = nir_load_subgroup_invocation(&b);
   nir_def *offset = nir_imul_imm(&b, lane, 4);
   nir_store_shared(&b, lane, offset, .write_mask = 1, .align_mul = 4);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = nir_memory_semantics(NIR_MEMORY_ACQUIRE |
                                                        NIR_MEMORY_RELEASE),
               .memory_modes = nir_var_mem_shared);
   nir_def *read = nir_load_shared(&b, 1, 32, offset, .align_mul = 4);
   nir_store_ssbo(&b, read, nir_imm_int(&b, 0), offset, .write_mask = 1,
                  .align_mul = 4);
   agx_shader_part compiled = {};
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, profile, reason);
   free(compiled.binary);
   ralloc_free(b.shader);
   return ok;
}

TEST(Apple9SharedMemory, NativeCeilingAcceptsThirtyTwoKiB)
{
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   EXPECT_FALSE(compile_shared(65536, &profile, &reason));
   EXPECT_NE(reason, nullptr);
   EXPECT_TRUE(compile_shared(32768, &profile, &reason));
   EXPECT_EQ(profile.required_threadgroup_memory_bytes, 32768u);
}

TEST(Apple9SharedMemory, RejectedAllocationDoesNotChangeFollowingCompiles)
{
   for (unsigned bytes : {16384u, 32768u}) {
      agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(compile_shared(bytes, &profile, &reason))
         << (reason ? reason : "no diagnostic");
      EXPECT_EQ(profile.required_threadgroup_memory_bytes, bytes);
      EXPECT_TRUE(agx_apple9_launch_threadgroup_memory_supported(
         profile.required_threadgroup_memory_bytes));
      EXPECT_FALSE(compile_shared(65536, &profile, &reason));
   }
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   EXPECT_TRUE(compile_shared(16384, &profile, &reason));
   EXPECT_EQ(profile.required_threadgroup_memory_bytes, 16384u);
}

TEST(Apple9SharedMemory, RequiredBytesRoundToThePowerOfTwo)
{
   const unsigned sizes[] = {8193, 12000, 16384, 16385, 32768};
   const unsigned expected[] = {16384, 16384, 16384, 32768, 32768};
   for (unsigned i = 0; i < 5; ++i) {
      SCOPED_TRACE(sizes[i]);
      agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(compile_shared(sizes[i], &profile, &reason))
         << (reason ? reason : "no diagnostic");
      EXPECT_EQ(profile.required_threadgroup_memory_bytes, expected[i]);
      EXPECT_TRUE(agx_apple9_launch_threadgroup_memory_supported(
         profile.required_threadgroup_memory_bytes));
      EXPECT_EQ(profile.resource_kind[profile.resource_binding_count - 1],
                AGX_APPLE9_COMPUTE_RESOURCE_SHARED);
   }
}
