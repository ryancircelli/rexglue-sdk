/**
 * @file        audio/downmix.cpp
 * @brief       Output-stage mix parameters: 5.1 fold, 5.1 matrix and master gain
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
#include <algorithm>
#include <atomic>
#include <mutex>

#include <rex/audio/downmix.h>

namespace rex::audio {
namespace {

// One output device exists, so the mix parameters are process state. The
// reader is the SDL device callback, which runs every 5.33 ms, so a plain
// mutex costs nothing and avoids a torn read across the four weights.
std::mutex g_mutex;
StereoFold g_fold = {};
SurroundMix g_mix = {};
float g_gain = 1.0f;

}  // namespace

void SetStereoFold(const StereoFold& fold) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_fold = fold;
}

StereoFold GetStereoFold() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_fold;
}

void SetSurroundMix(const SurroundMix& mix) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_mix = mix;
}

SurroundMix GetSurroundMix() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_mix;
}

void SetOutputGain(float linear) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_gain = linear;
}

float GetOutputGain() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_gain;
}

namespace {

// Headphone model at 48 kHz. Interaural delays from the Woodworth formula (head radius 8.75 cm):
// 30 degrees -> 0.26 ms, 110 degrees (70 degrees lateral) -> 0.55 ms.
constexpr size_t kFrontFarDelay = 13;
constexpr size_t kRearFarDelay = 26;
// One-pole low-pass coefficients, a = 1 - exp(-2 pi fc / 48000).
constexpr float kFrontFarLowPass = 0.480f;  // 5 kHz head shadow
constexpr float kRearNearLowPass = 0.692f;  // 9 kHz, rear (pinna) cue
constexpr float kRearFarLowPass = 0.279f;   // 2.5 kHz, deeper shadow
// Ear gains: the far ear is quieter, more so for the surrounds.
constexpr float kFrontFarGain = 0.75f;
constexpr float kRearNearGain = 0.90f;
constexpr float kRearFarGain = 0.55f;
constexpr float kCenterGain = 0.70710678f;
// LFE is authored 10 dB hot for a subwoofer; headphones get some of it back.
constexpr float kLfeGain = 0.5f;
// Keeps the sum of six channels (two of them doubled) out of clipping.
constexpr float kOutputScale = 0.55f;

std::atomic<float> g_latency_ms{0.0f};
std::atomic<uint64_t> g_underruns{0};

}  // namespace

void HeadphoneVirtualizer::Process(const float* in, float* out, size_t frames) {
  constexpr size_t kMask = kHistory - 1;
  for (size_t i = 0; i < frames; ++i) {
    const float* s = in + i * 6;
    const float fl = s[0], fr = s[1], fc = s[2], lf = s[3], bl = s[4], br = s[5];
    history_[0][pos_] = fl;
    history_[1][pos_] = fr;
    history_[2][pos_] = bl;
    history_[3][pos_] = br;
    const size_t front_far = (pos_ - kFrontFarDelay) & kMask;
    const size_t rear_far = (pos_ - kRearFarDelay) & kMask;
    pos_ = (pos_ + 1) & kMask;

    // Far ear of each front speaker: delayed and low-passed.
    front_far_[0] += kFrontFarLowPass * (history_[0][front_far] - front_far_[0]);
    front_far_[1] += kFrontFarLowPass * (history_[1][front_far] - front_far_[1]);
    // Surrounds: duller at the near ear, more delayed and duller at the far ear.
    rear_near_[0] += kRearNearLowPass * (bl - rear_near_[0]);
    rear_near_[1] += kRearNearLowPass * (br - rear_near_[1]);
    rear_far_[0] += kRearFarLowPass * (history_[2][rear_far] - rear_far_[0]);
    rear_far_[1] += kRearFarLowPass * (history_[3][rear_far] - rear_far_[1]);

    const float mid = fc * kCenterGain + lf * kLfeGain;
    const float left = fl + kFrontFarGain * front_far_[1] + kRearNearGain * rear_near_[0] +
                       kRearFarGain * rear_far_[1] + mid;
    const float right = fr + kFrontFarGain * front_far_[0] + kRearNearGain * rear_near_[1] +
                        kRearFarGain * rear_far_[0] + mid;
    out[i * 2] = std::clamp(left * kOutputScale, -1.0f, 1.0f);
    out[i * 2 + 1] = std::clamp(right * kOutputScale, -1.0f, 1.0f);
  }
}

void Upmix51To71(const float* in, float* out, size_t frames) {
  constexpr float kSpread = 0.70710678f;
  for (size_t i = 0; i < frames; ++i) {
    const float* s = in + i * 6;
    float* d = out + i * 8;
    d[0] = s[0];
    d[1] = s[1];
    d[2] = s[2];
    d[3] = s[3];
    d[4] = s[4] * kSpread;  // back left
    d[5] = s[5] * kSpread;  // back right
    d[6] = s[4] * kSpread;  // side left
    d[7] = s[5] * kSpread;  // side right
  }
}

void SetOutputLatencyMs(float ms) {
  g_latency_ms.store(ms, std::memory_order_relaxed);
}

float GetOutputLatencyMs() {
  return g_latency_ms.load(std::memory_order_relaxed);
}

void CountOutputUnderrun() {
  g_underruns.fetch_add(1, std::memory_order_relaxed);
}

uint64_t GetOutputUnderruns() {
  return g_underruns.load(std::memory_order_relaxed);
}

}  // namespace rex::audio
