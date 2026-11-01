/**
 * @file        audio/downmix.h
 * @brief       Output-stage mix parameters: 5.1 fold, 5.1 matrix and master gain
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace rex::audio {

/**
 * Weights applied when the guest's 5.1 render is folded to a stereo device.
 * Guest channel order is the XAudio default: fl fr fc lf bl br. The front
 * channels carry an implicit weight of 1.0, so `scale` alone sets the output
 * level.
 *
 * Defaults follow ITU-R BS.775 and Dolby Lo/Ro: center and surround at -3 dB,
 * LFE dropped. No published stereo downmix folds LFE, which is authored around
 * 10 dB hot by convention, so a title that wants it audible sets a weight that
 * undoes that offset rather than folding it at unity.
 */
struct StereoFold {
  float center = 0.70710678f;
  float surround = 0.70710678f;
  float lfe = 0.0f;
  float scale = 0.58578644f;  // 1/(1+0.707)
};

/**
 * Weights applied when the guest's 5.1 render reaches a device with more than
 * two channels, which the output stage passes through rather than folding.
 * Same channel order and the same implicit front weight of 1.0 as StereoFold,
 * so the two structs describe one mix in two destinations.
 *
 * Defaults are unity, a passthrough, because a title whose render really is
 * 5.1 wants its own mix reproduced. A title that packs something other than an
 * LFE into the LFE slot sets `lfe` to drop or attenuate it: that slot is
 * reproduced around 10 dB hot, so full-band content placed there arrives far
 * louder than it was authored, and a downstream downmix that folds it without
 * low-passing carries the whole band into the other channels.
 */
struct SurroundMix {
  float center = 1.0f;
  float surround = 1.0f;
  float lfe = 1.0f;
};

/// Safe to call from any thread. The output stage picks the new values up on
/// its next device callback.
void SetStereoFold(const StereoFold& fold);
StereoFold GetStereoFold();

void SetSurroundMix(const SurroundMix& mix);
SurroundMix GetSurroundMix();

/// Linear master gain applied to the stereo fold and to 5.1 passthrough alike.
/// 1.0 is unity. Above unity can clip, and the output stage clamps.
void SetOutputGain(float linear);
float GetOutputGain();

/// Speaker layout of the output stage (audio_output option).
enum class OutputLayout { kStereo, kHeadphones, kSurround51, kSurround71 };

/**
 * Binaural rendering of the guest 5.1 mix for headphones. Every speaker becomes a virtual source at
 * its standard angle (front 30 degrees, center 0, surround 110): the far ear hears it later
 * (interaural time difference, Woodworth model) and duller (head shadow low-pass), and the
 * surrounds lose some treble at both ears (rear cue). Stateful: one instance per output stream.
 */
class HeadphoneVirtualizer {
 public:
  /// in: interleaved fl fr fc lf bl br; out: interleaved l r, clamped to [-1, 1].
  void Process(const float* in, float* out, size_t frames);

 private:
  static constexpr size_t kHistory = 64;  // longer than the longest delay, power of two
  float history_[4][kHistory] = {};       // fl fr bl br
  size_t pos_ = 0;
  float front_far_[2] = {};  // fl -> right ear, fr -> left ear
  float rear_near_[2] = {};  // bl -> left ear, br -> right ear
  float rear_far_[2] = {};   // bl -> right ear, br -> left ear
};

/// 5.1 -> 7.1 (SDL/Windows order fl fr fc lf bl br sl sr): each 5.1 surround is spread over the
/// side and back speakers at -3 dB, keeping its power and placing it between them (about 110
/// degrees, where 5.1 surrounds belong) instead of only behind the listener.
void Upmix51To71(const float* in, float* out, size_t frames);

/// Output latency estimate in milliseconds: guest frames waiting to play plus the audio queued in
/// the output stream and the device buffer. Updated by the output stage on every callback.
void SetOutputLatencyMs(float ms);
float GetOutputLatencyMs();

/// Device callbacks that found no guest audio to play (silence was output instead: a gap). Counts
/// from startup; the game itself sends silence frames when nothing plays, so only a rising count
/// during gameplay means the queue ran dry.
void CountOutputUnderrun();
uint64_t GetOutputUnderruns();

}  // namespace rex::audio
