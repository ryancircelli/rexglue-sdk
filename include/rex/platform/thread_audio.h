/**
 * @file        platform/thread_audio.h
 * @brief       Scheduling class for the threads that produce sound
 *
 * Kept out of rex/platform/thread.h on purpose: that header is in the recompiled game's
 * precompiled header, and a declaration added there rebuilds all of the recompiled code.
 */

#pragma once

#include <string>
#include <string_view>

#include <rex/platform.h>

namespace rex::platform {

class Thread;

/**
 * Puts a thread (null = the calling one) in the scheduling class for real-time audio work: the
 * threads that produce each sound block before the device asks for it.
 *
 * @param policy realtime = Mach time-constraint policy (5.33 ms period, 2 ms computation), as Core
 *               Audio's own I/O thread; fifo = SCHED_FIFO priority 47; anything else leaves the
 *               thread as it is. A no-op where the platform has no such class.
 * @return What was done, for the log.
 */
REX_API std::string SetThreadAudioPriority(Thread* thread, std::string_view policy);

}  // namespace rex::platform
