/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include <rex/assert.h>
#include <rex/audio/conversion.h>
#include <rex/audio/downmix.h>
#include <rex/audio/flags.h>
#include <rex/audio/sdl/sdl_audio_driver.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/logging.h>
#include <rex/perf/counter.h>
#include <SDL3/SDL.h>

REXCVAR_DEFINE_BOOL(audio_mute, false, "Audio", "Mute audio output");

REXCVAR_DEFINE_STRING(audio_output, "auto", "Audio",
                      "Speaker layout: auto (follows the Windows device: stereo, 5.1 or 7.1), "
                      "stereo, headphones (virtual surround: the 5.1 mix rendered for "
                      "headphones), 5.1 or 7.1")
    .allowed({"auto", "stereo", "headphones", "5.1", "7.1"})
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

namespace rex::audio::sdl {

SDLAudioDriver::SDLAudioDriver(memory::Memory* memory, rex::thread::Semaphore* semaphore)
    : AudioDriver(memory), semaphore_(semaphore) {}

SDLAudioDriver::~SDLAudioDriver() {
  assert_true(frames_queued_.empty());
  assert_true(frames_unused_.empty());
}

bool SDLAudioDriver::Initialize() {
  // Set audio category for proper OS audio handling
  SDL_SetHint(SDL_HINT_AUDIO_CATEGORY, "playback");

  // Set app name for audio device identification
  SDL_SetAppMetadataProperty(SDL_PROP_APP_METADATA_NAME_STRING, "rexglue");

  if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
    REXAPU_ERROR("SDL_InitSubSystem(SDL_INIT_AUDIO) failed: {}", SDL_GetError());
    return false;
  }
  sdl_initialized_ = true;

  // The guest always renders 5.1. The stream is opened with the channel count of the chosen
  // layout; SDL converts it to the device's own layout when they differ.
  if (!OpenStream(frame_channels_)) {
    return false;
  }
  SDL_AudioDeviceID sdl_device = SDL_GetAudioStreamDevice(sdl_stream_);
  if (!sdl_device) {
    REXAPU_ERROR("SDL_GetAudioStreamDevice() failed: {}", SDL_GetError());
    return false;
  }

  SDL_AudioSpec obtained_spec = {};
  if (!SDL_GetAudioDeviceFormat(sdl_device, &obtained_spec, &device_buffer_frames_)) {
    REXAPU_WARN("SDL_GetAudioDeviceFormat() failed: {}", SDL_GetError());
    obtained_spec.freq = frame_frequency_;
    obtained_spec.format = SDL_AUDIO_F32LE;
    obtained_spec.channels = frame_channels_;
    device_buffer_frames_ = 0;
  }

  const std::string& option = REXCVAR_GET(audio_output);
  if (option == "stereo") {
    layout_ = OutputLayout::kStereo;
  } else if (option == "headphones") {
    layout_ = OutputLayout::kHeadphones;
  } else if (option == "5.1") {
    layout_ = OutputLayout::kSurround51;
  } else if (option == "7.1") {
    layout_ = OutputLayout::kSurround71;
  } else {
    // auto: follow the device. A 1-channel device gets the stereo fold too, then SDL collapses
    // it to mono.
    layout_ = obtained_spec.channels >= 8   ? OutputLayout::kSurround71
              : obtained_spec.channels >= 6 ? OutputLayout::kSurround51
                                            : OutputLayout::kStereo;
  }
  const int channels = layout_ == OutputLayout::kSurround71   ? 8
                       : layout_ == OutputLayout::kSurround51 ? 6
                                                              : 2;
  if (channels != static_cast<int>(frame_channels_)) {
    SDL_DestroyAudioStream(sdl_stream_);
    sdl_stream_ = nullptr;
    if (!OpenStream(channels)) {
      return false;
    }
    sdl_device = SDL_GetAudioStreamDevice(sdl_stream_);
    if (!sdl_device) {
      REXAPU_ERROR("SDL_GetAudioStreamDevice() failed: {}", SDL_GetError());
      return false;
    }
  }

  // The endpoint layout decides which mix the callback runs, and it is the
  // first thing worth knowing when a report says the balance is wrong on one
  // speaker setup and right on another.
  static constexpr const char* kLayoutNames[] = {"stereo", "headphones (virtual surround)",
                                                 "5.1", "7.1"};
  const char* device_name = SDL_GetAudioDeviceName(sdl_device);
  REXAPU_INFO(
      "audio endpoint '{}': {} ch, {} Hz, format 0x{:04X}, buffer {} frames; output {} ({} ch)",
      device_name ? device_name : "?", obtained_spec.channels, obtained_spec.freq,
      static_cast<uint32_t>(obtained_spec.format), device_buffer_frames_,
      kLayoutNames[static_cast<int>(layout_)], static_cast<int>(sdl_device_channels_));

  if (!SDL_ResumeAudioDevice(sdl_device)) {
    REXAPU_ERROR("SDL_ResumeAudioDevice() failed: {}", SDL_GetError());
    return false;
  }

  return true;
}

bool SDLAudioDriver::OpenStream(int channels) {
  SDL_AudioSpec desired_spec = {};
  desired_spec.freq = frame_frequency_;
  desired_spec.format = SDL_AUDIO_F32LE;
  desired_spec.channels = channels;
  sdl_device_channels_ = static_cast<uint8_t>(channels);
  sdl_stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &desired_spec,
                                          SDLCallback, this);
  if (!sdl_stream_) {
    REXAPU_ERROR("SDL_OpenAudioDeviceStream({} ch) failed: {}", channels, SDL_GetError());
    return false;
  }
  return true;
}

void SDLAudioDriver::SubmitFrame(uint32_t frame_ptr) {
  const auto input_frame = memory_->TranslateVirtual<float*>(frame_ptr);
  float* output_frame;
  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    if (frames_unused_.empty()) {
      output_frame = new float[frame_samples_];
    } else {
      output_frame = frames_unused_.top();
      frames_unused_.pop();
    }
  }

  std::memcpy(output_frame, input_frame, frame_samples_ * sizeof(float));

  static uint32_t sdl_submit_count = 0;
  if (sdl_submit_count < 10) {
    REXAPU_DEBUG("SDLAudioDriver::SubmitFrame: frame_ptr={:08X} queued_count={}", frame_ptr,
                 frames_queued_.size() + 1);
    sdl_submit_count++;
  }

  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    frames_queued_.push(output_frame);
    PROFILE_BUFFER_QUEUE_DEPTH(static_cast<int64_t>(frames_queued_.size()));
  }
}

void SDLAudioDriver::Shutdown() {
  if (sdl_stream_) {
    SDL_DestroyAudioStream(sdl_stream_);
    sdl_stream_ = nullptr;
  }
  if (sdl_initialized_) {
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    sdl_initialized_ = false;
  }
  std::unique_lock<std::mutex> guard(frames_mutex_);
  while (!frames_unused_.empty()) {
    delete[] frames_unused_.top();
    frames_unused_.pop();
  }
  while (!frames_queued_.empty()) {
    delete[] frames_queued_.front();
    frames_queued_.pop();
  }
}

void SDLAudioDriver::SDLCallback(void* userdata, SDL_AudioStream* stream, int additional_amount,
                                 [[maybe_unused]] int total_amount) {
  SCOPE_profile_cpu_f("apu");
  if (!userdata || !stream) {
    REXAPU_ERROR("SDLAudioDriver::SDLCallback called with nullptr.");
    return;
  }
  const auto driver = static_cast<SDLAudioDriver*>(userdata);
  const int sample_count =
      static_cast<int>(channel_samples_ * std::max<uint8_t>(driver->sdl_device_channels_, 1));
  const int len = static_cast<int>(sizeof(float) * sample_count);
  float* data = SDL_stack_alloc(float, sample_count);
  // 5.1 intermediate for the layouts that are built from it (7.1 and headphones).
  float* surround = SDL_stack_alloc(float, channel_samples_ * frame_channels_);
  if (!data || !surround) {
    REXAPU_ERROR("SDLAudioDriver::SDLCallback failed to allocate {} samples", sample_count);
    if (data) SDL_stack_free(data);
    if (surround) SDL_stack_free(surround);
    return;
  }
  {
    // Latency estimate: guest frames waiting + audio queued in the stream + device buffer.
    size_t queued_frames;
    {
      std::unique_lock<std::mutex> guard(driver->frames_mutex_);
      queued_frames = driver->frames_queued_.size();
    }
    const int bytes_per_frame = static_cast<int>(sizeof(float)) * driver->sdl_device_channels_;
    const int stream_frames = bytes_per_frame ? SDL_GetAudioStreamQueued(stream) / bytes_per_frame
                                              : 0;
    const double samples = double(queued_frames) * channel_samples_ +
                           double(std::max(stream_frames, 0)) +
                           double(driver->device_buffer_frames_);
    SetOutputLatencyMs(static_cast<float>(samples * 1000.0 / frame_frequency_));
  }
  // Snapshot once. A change mid-callback would split the frame across two mixes.
  const StereoFold fold = GetStereoFold();
  const SurroundMix mix = GetSurroundMix();
  const float gain = GetOutputGain();
  while (additional_amount > 0) {
    static uint32_t sdl_callback_count = 0;
    std::unique_lock<std::mutex> guard(driver->frames_mutex_);
    if (driver->frames_queued_.empty()) {
      if (sdl_callback_count < 10) {
        REXAPU_DEBUG("SDLCallback: no frames queued (silence)");
        sdl_callback_count++;
      }
      CountOutputUnderrun();
      std::memset(data, 0, len);
      if (!SDL_PutAudioStreamData(stream, data, len)) {
        REXAPU_ERROR("SDL_PutAudioStreamData() failed while filling silence: {}", SDL_GetError());
        break;
      }
      additional_amount -= len;
    } else {
      auto buffer = driver->frames_queued_.front();
      driver->frames_queued_.pop();
      if (REXCVAR_GET(audio_mute)) {
        std::memset(data, 0, len);
      } else {
        switch (driver->layout_) {
          case OutputLayout::kStereo:
            conversion::sequential_6_BE_to_interleaved_2_LE(data, buffer, channel_samples_, fold,
                                                            gain);
            break;
          case OutputLayout::kHeadphones:
            conversion::sequential_6_BE_to_interleaved_6_LE(surround, buffer, channel_samples_,
                                                            SurroundMix{}, gain);
            driver->headphones_.Process(surround, data, channel_samples_);
            break;
          case OutputLayout::kSurround51:
            conversion::sequential_6_BE_to_interleaved_6_LE(data, buffer, channel_samples_, mix,
                                                            gain);
            break;
          case OutputLayout::kSurround71:
            conversion::sequential_6_BE_to_interleaved_6_LE(surround, buffer, channel_samples_,
                                                            mix, gain);
            Upmix51To71(surround, data, channel_samples_);
            break;
        }
      }
      if (!SDL_PutAudioStreamData(stream, data, len)) {
        REXAPU_ERROR("SDL_PutAudioStreamData() failed: {}", SDL_GetError());
        driver->frames_unused_.push(buffer);
        break;
      }
      driver->frames_unused_.push(buffer);

      auto ret = driver->semaphore_->Release(1, nullptr);
      assert_true(ret);
      additional_amount -= len;
    }
  }
  SDL_stack_free(surround);
  SDL_stack_free(data);
}

}  // namespace rex::audio::sdl
