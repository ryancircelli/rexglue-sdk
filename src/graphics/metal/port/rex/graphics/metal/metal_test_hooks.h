#pragma once
// Test-only injection points for the texture cache stress test
// (src/graphics/metal/tests/texcache_stress.cc). They exist only when the
// sources are compiled with REX_METAL_TEST_HOOKS=1 (the test target); in the
// plugin every hook expands to nothing.

#if defined(REX_METAL_TEST_HOOKS) && REX_METAL_TEST_HOOKS

#include <cstdint>

namespace rex::graphics::metal::test_hooks {

enum class Point : uint32_t {
  // LoadTextureDataFromCpuGuestMemory: the load watch is armed (watch-first
  // order) or not (old order); guest RAM not copied yet.
  kCpuSourceBeforeCopy,
  // LoadTextureDataFromCpuGuestMemory: guest RAM copied, the texture not yet
  // marked up to date and watched.
  kCpuSourceAfterCopy,
  // LoadTextureDataFromResidentMemoryImpl: the range is valid in shared
  // memory, the copy from the shared memory buffer not encoded yet.
  kResidentBeforeCopy,
  // LoadTextureDataFromResidentMemoryImpl: the copy is encoded, the texture
  // not yet marked up to date and watched.
  kResidentAfterCopy,
};

// base_page: the texture key's physical 4 KB page of mip 0.
using HookFn = void (*)(Point point, uint32_t base_page);
inline HookFn g_hook = nullptr;

}  // namespace rex::graphics::metal::test_hooks

#define REX_METAL_TEST_HOOK(point, texture)                                                        \
  do {                                                                                             \
    if (::rex::graphics::metal::test_hooks::g_hook) {                                              \
      ::rex::graphics::metal::test_hooks::g_hook(::rex::graphics::metal::test_hooks::Point::point, \
                                                 (texture).key().base_page);                       \
    }                                                                                              \
  } while (0)
#define REX_METAL_TEST_FRIEND        \
  friend class TexCacheStressAccess; \
  friend class TexConfAccess;

#else

#define REX_METAL_TEST_HOOK(point, texture) ((void)0)
#define REX_METAL_TEST_FRIEND

#endif
