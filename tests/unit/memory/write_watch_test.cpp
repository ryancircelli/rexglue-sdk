/**
 * @file        write_watch_test.cpp
 * @brief       Write watches across host pages that hold several guest pages
 *
 * On macOS arm64 the host page is 16 KiB while the guest page is 4 KiB, so a
 * host page can hold several guest pages with different protections. A watch
 * has to take the access of every guest page in the host page, both when it is
 * armed and when the faulting page is unprotected again.
 *
 * @copyright   Copyright (c) 2026 Tom Clay
 * @license     BSD 3-Clause License
 */

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <utility>

#include <rex/memory.h>
#include <rex/system/xmemory.h>
#include <rex/thread/mutex.h>

#include "test_memory.h"

namespace {

std::atomic<int> g_watch_hits{0};

std::pair<uint32_t, uint32_t> CountWatchHit(void*, uint32_t, uint32_t, bool) {
  g_watch_hits.fetch_add(1);
  return {0, UINT32_MAX};
}

}  // namespace

TEST_CASE("A write to a watched host page unprotects it by the access of every guest page",
          "[memory][watch]") {
  const uint32_t host_page = uint32_t(rex::memory::page_size());
  constexpr uint32_t kGuestPage = 0x1000;
  if (host_page <= kGuestPage) {
    SKIP("the host page is not larger than a guest page");
  }
  auto& memory = rex::testing::GetTestMemory();
  const uint32_t address =
      memory.SystemHeapAlloc(host_page, host_page, rex::memory::kSystemHeapPhysical);
  REQUIRE(address != 0);
  auto* host = memory.TranslateVirtual<uint8_t*>(address);
  REQUIRE((reinterpret_cast<uintptr_t>(host) & (host_page - 1)) == 0);
  auto* heap = memory.LookupHeap(address);
  REQUIRE(heap != nullptr);
  void* handle = nullptr;
  const auto cleanup = [&] {
    heap->Protect(address, host_page,
                  rex::memory::kMemoryProtectRead | rex::memory::kMemoryProtectWrite);
    if (handle) {
      memory.UnregisterPhysicalMemoryInvalidationCallback(handle);
    }
    memory.SystemHeapFree(address);
  };

  // The first guest page of the host page is read only, the second one stays
  // writable - the host page as a whole is writable.
  REQUIRE(heap->Protect(address, kGuestPage, rex::memory::kMemoryProtectRead));

  handle = memory.RegisterPhysicalMemoryInvalidationCallback(CountWatchHit, nullptr);
  g_watch_hits = 0;
  memory.EnablePhysicalMemoryAccessCallbacks(memory.GetPhysicalAddress(address), host_page, true,
                                             false);
  size_t length = host_page;
  rex::memory::PageAccess access = rex::memory::PageAccess::kReadWrite;
  REQUIRE(rex::memory::QueryProtect(host, length, access));
  CHECK(access == rex::memory::PageAccess::kReadOnly);

  // A write to the second guest page of the host page fires the watch and
  // makes the whole host page writable again.
  {
    auto global_lock = rex::thread::global_critical_region::AcquireDirect();
    CHECK(memory.TriggerPhysicalMemoryCallbacks(std::move(global_lock), address + kGuestPage, 1,
                                                true, false));
  }
  CHECK(g_watch_hits.load() == 1);
  length = host_page;
  REQUIRE(rex::memory::QueryProtect(host, length, access));
  CHECK(access == rex::memory::PageAccess::kReadWrite);

  cleanup();
}
