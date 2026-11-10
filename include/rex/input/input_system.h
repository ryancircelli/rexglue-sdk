#pragma once
/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <array>
#include <atomic>
#include <bitset>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <rex/input/device_assignment.h>
#include <rex/input/input.h>
#include <rex/input/input_driver.h>
#include <rex/input/slot_table.h>
#include <rex/input/state_merge.h>
#include <rex/system/interfaces/input.h>

namespace rex::ui {
class Window;
}

namespace rex::input {

class InputSystem : public system::IInputSystem {
 public:
  explicit InputSystem(rex::ui::Window* window);
  ~InputSystem() override;

  rex::ui::Window* window() const { return window_; }

  X_STATUS Setup() override;
  void Shutdown() override;

  void AddDriver(std::unique_ptr<InputDriver> driver);
  void AttachWindow(rex::ui::Window* window);
  void SetActiveCallback(std::function<bool()> callback);

  /// Replaces any previous assignment. Call before the guest starts polling.
  void SetDeviceAssignment(std::unique_ptr<DeviceAssignment> assignment);

  /// True once after a controller connected or disconnected (or moved to another player), so the
  /// caller can tell the guest (XN_SYS_INPUTDEVICESCHANGED) outside the input lock.
  bool TakeDevicesChanged() { return devices_changed_.exchange(false, std::memory_order_acq_rel); }

  X_RESULT GetCapabilities(uint32_t user_index, uint32_t flags, X_INPUT_CAPABILITIES* out_caps);
  X_RESULT GetState(uint32_t user_index, X_INPUT_STATE* out_state);
  /// GetState for the emulator's own UI, which reads while the guest is blocked.
  X_RESULT GetStateForUI(uint32_t user_index, X_INPUT_STATE* out_state);
  X_RESULT SetState(uint32_t user_index, X_INPUT_VIBRATION* vibration);
  X_RESULT GetKeystroke(uint32_t user_index, uint32_t flags, X_INPUT_KEYSTROKE* out_keystroke);

  /// While any blocker is held the guest reads a neutral pad. Buttons still
  /// held when the last one drops stay masked until released, so the press
  /// that dismissed the dialog does not also reach the game.
  void AddUIInputBlocker();
  void RemoveUIInputBlocker();

  bool GetVibrationEnabled() const;
  void ToggleVibration();

  std::bitset<kMaxGuestUsers> GetConnectedUsers() const { return connected_users_; }
  /// Guest user whose device most recently produced a button press.
  uint32_t GetLastUsedUser() const { return last_used_user_; }

  /// One device as the guest sees it, for diagnostics (rex-padtest, the state
  /// log).
  struct DeviceSnapshot {
    DeviceInfo info;
    int32_t user = -1;  // guest user it feeds, -1 for none
    X_INPUT_STATE state = {};
    bool has_state = false;
    X_INPUT_CAPABILITIES caps = {};
    bool has_caps = false;
  };
  /// Re-enumerates and reads every device's raw state (no deadzones, no UI
  /// blockers).
  std::vector<DeviceSnapshot> SnapshotDevices();

  /// With input_log_state on, logs every pad's state at most every 0.5 s.
  /// GetState calls it; tools that only take snapshots call it themselves.
  void LogStateIfDue();
  /// Guest user a device feeds, or -1.
  int32_t UserForDevice(DeviceId id) const;
  /// Rumbles one device directly, bypassing the guest user mapping.
  X_RESULT SetDeviceVibration(DeviceId id, X_INPUT_VIBRATION* vibration);
  const SlotTable& slot_table() const { return slot_table_; }

 private:
  std::atomic<bool> devices_changed_{false};
  using JoystickValue = std::pair<uint16_t, uint16_t>;
  /// Re-enumerates every driver and notifies the assignment when the set
  /// changed.
  void RefreshDevices();
  /// Reads the slot cvars into slot_table_.
  void ApplySlotConfig();
  /// input_log_state: logs every device every 0.5 s.
  InputDriver* DriverForDevice(DeviceId id);
  const DeviceInfo* DeviceInfoFor(DeviceId id) const;
  /// The device that speaks for a user, preferring the one most recently in
  /// the player's hands.
  DeviceId ChooseDeviceForUser(uint32_t user_index) const;

  void UpdateConnectedUser(uint32_t user_index, bool connected);
  void AdjustDeadzoneLevels(uint32_t user_index, X_INPUT_GAMEPAD* gamepad) const;
  X_INPUT_VIBRATION ModifyVibrationLevel(const X_INPUT_VIBRATION* vibration) const;

  rex::ui::Window* window_ = nullptr;

  std::vector<std::unique_ptr<InputDriver>> drivers_;

  std::unique_ptr<DeviceAssignment> assignment_;
  ActiveDeviceTracker active_devices_;

  // Ordered by ordinal. Ordinals are never recycled, so unplugging pad one
  // does not renumber pad two.
  std::vector<DeviceInfo> devices_;
  std::vector<InputDriver*> device_owners_;

  std::bitset<kMaxGuestUsers> connected_users_;
  uint32_t last_used_user_ = 0;
  // {left, right} stick maxima, scaling the deadzone percentages.
  std::array<std::pair<JoystickValue, JoystickValue>, kMaxGuestUsers> user_max_joystick_value_ = {};

  std::atomic<int> ui_input_blockers_{0};
  // Masked out per user until the guest sees them released.
  std::array<uint16_t, kMaxGuestUsers> consumed_buttons_ = {};

  SlotTable slot_table_;
  std::string applied_pins_text_;
  std::set<DeviceId> ignored_virtual_;
  bool steam_virtual_warned_ = false;
  std::atomic<int64_t> last_state_log_ms_{0};
};

/// The scripted virtual pads the hid_virtual_pad* cvars ask for, or null.
std::unique_ptr<InputDriver> CreateScriptPadDriverFromConfig();

/// Create a default InputSystem with SDL + MnK + NOP drivers.
/// In tool mode, only the NOP driver is added.
std::unique_ptr<InputSystem> CreateDefaultInputSystem(bool tool_mode);

}  // namespace rex::input
