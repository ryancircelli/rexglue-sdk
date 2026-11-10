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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>

#include <rex/dbg.h>
#include <rex/input/device_assignment.h>
#include <rex/input/flags.h>
#include <rex/input/input_driver.h>
#include <rex/input/input_system.h>
#include <rex/input/mnk/mnk_input_driver.h>
#include <rex/input/nop/nop_input_driver.h>
#include <rex/input/script/script_input_driver.h>
#include <rex/input/sdl/sdl_input_driver.h>
#include <rex/input/state_log.h>
#include <rex/input/state_merge.h>
#include <rex/input/steam_detect.h>
#include <rex/input/xinput/xinput_input_driver.h>
#include <rex/logging.h>
#include <rex/system/kernel_state.h>

REXCVAR_DEFINE_STRING(input_backend, "sdl", "Input", "Input backend: sdl, xinput")
    .allowed({"sdl", "xinput"});

REXCVAR_DEFINE_BOOL(guide_button, false, "Input", "Enable guide button pass-through");

REXCVAR_DEFINE_BOOL(vibration, true, "Input", "Enable controller vibration");

REXCVAR_DEFINE_DOUBLE(left_stick_deadzone_percentage, 0.0, "Input",
                      "Deadzone applied to the left stick, as a fraction of its range")
    .range(0.0, 1.0);
REXCVAR_DEFINE_DOUBLE(right_stick_deadzone_percentage, 0.0, "Input",
                      "Deadzone applied to the right stick, as a fraction of its range")
    .range(0.0, 1.0);

REXCVAR_DEFINE_STRING(input_player_slots, "", "Input",
                      "Pin controllers to players, e.g. \"1=<serial>;2=054c:0ce6;3=Xbox\": each "
                      "player 1-4 takes the pad whose serial, GUID or vid:pid equals the text, or "
                      "whose name contains it. rex-padtest prints these for each pad. Pads not "
                      "pinned take players in connection order");
REXCVAR_DEFINE_STRING(input_virtual_pads, "after", "Input",
                      "Software-made pads (Steam Input's virtual pad): after = never player 1, "
                      "normal = like any pad, ignore = hidden from the game")
    .allowed({"after", "normal", "ignore"});
REXCVAR_DEFINE_BOOL(input_remember_slots, true, "Input",
                    "A controller that disconnects gets its player back when it reconnects");
REXCVAR_DEFINE_BOOL(input_log_state, false, "Input",
                    "Log every controller's player and state every 0.5 s (env REX_INPUT_LOG_STATE=1)");

REXCVAR_DEFINE_INT32(hid_virtual_pads, 0, "Input",
                     "Scripted virtual pads on players 1-N, for testing split screen without "
                     "controllers. Each replays hid_virtual_pad1..4 or the matching line of "
                     "hid_virtual_pad_script; a pad without a script stays connected and idle")
    .range(0, 4);
REXCVAR_DEFINE_STRING(hid_virtual_pad_script, "", "Input",
                      "File of virtual pad scripts: \"pad1 = 25:down,27:a\" lines up to pad4 "
                      "(a bo2 .route file works as is). Sets the pad count when "
                      "hid_virtual_pads is 0");
REXCVAR_DEFINE_STRING(hid_virtual_pad1, "", "Input",
                      "Virtual pad 1 script: \"25:a,60-90:stick_up+look_right,100-105:disconnect\" "
                      "(seconds from startup). \"wait_event <name> [timeout_s]\" or "
                      "\"wait_log <text> [timeout_s]\" waits for the game, and later times count "
                      "from there. Overrides the script file");
REXCVAR_DEFINE_STRING(hid_virtual_pad2, "", "Input", "Virtual pad 2 script (see hid_virtual_pad1)");
REXCVAR_DEFINE_STRING(hid_virtual_pad3, "", "Input", "Virtual pad 3 script (see hid_virtual_pad1)");
REXCVAR_DEFINE_STRING(hid_virtual_pad4, "", "Input", "Virtual pad 4 script (see hid_virtual_pad1)");
REXCVAR_DEFINE_INT32(hid_virtual_pad_look_speed, 35, "Input",
                     "Right stick tilt of look_* in virtual pad scripts, in percent")
    .range(1, 100);

namespace rex::input {

namespace {

// Synthetic devices are parked past every physical ordinal so they cannot push
// a real pad off guest user 0. SlotAssignment routes them by their synthetic
// flag and never reads this value.
constexpr uint32_t kSyntheticOrdinal = UINT32_MAX;

// XN_SYS_INPUTDEVICESCHANGED
constexpr uint32_t kXNotificationSystemInputDevicesChanged = 0x00000012;

constexpr auto kStateLogInterval = std::chrono::milliseconds(500);

std::string DescribeDevice(const DeviceInfo& info) {
  std::string text = fmt::format("\"{}\"", info.name.empty() ? "?" : info.name);
  if (info.vendor_id || info.product_id) {
    text += fmt::format(" {:04x}:{:04x}", info.vendor_id, info.product_id);
  }
  text += " " + DeviceIdentityKey(info);
  if (info.virtual_device) {
    text += " (virtual)";
  }
  return text;
}

}  // namespace

InputSystem::InputSystem(rex::ui::Window* window) : window_(window) {}

InputSystem::~InputSystem() = default;

X_STATUS InputSystem::Setup() {
  return X_STATUS_SUCCESS;
}

X_RESULT InputSystem::SetDeviceVibration(DeviceId id, X_INPUT_VIBRATION* vibration) {
  auto* driver = DriverForDevice(id);
  if (!driver || !vibration) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  return driver->SetDeviceVibration(id, vibration);
}

void InputSystem::Shutdown() {
  // device_owners_ holds raw driver pointers.
  devices_.clear();
  device_owners_.clear();
  drivers_.clear();
}

void InputSystem::AddDriver(std::unique_ptr<InputDriver> driver) {
  drivers_.push_back(std::move(driver));
}

void InputSystem::AttachWindow(rex::ui::Window* window) {
  window_ = window;
  for (auto& driver : drivers_) {
    driver->OnWindowAvailable(window);
  }
}

void InputSystem::SetActiveCallback(std::function<bool()> callback) {
  for (auto& driver : drivers_) {
    driver->set_is_active_callback(callback);
  }
}

void InputSystem::SetDeviceAssignment(std::unique_ptr<DeviceAssignment> assignment) {
  assignment_ = std::move(assignment);
  if (assignment_) {
    assignment_->OnDevicesChanged(devices_);
  }
}

void InputSystem::ApplySlotConfig() {
  const std::string& pins_text = REXCVAR_GET(input_player_slots);
  if (pins_text != applied_pins_text_) {
    applied_pins_text_ = pins_text;
    std::string error;
    slot_table_.SetPins(ParseSlotPins(pins_text, &error));
    if (!error.empty()) {
      REXLOG_WARN("input_player_slots: {}", error);
    }
    for (const auto& pin : slot_table_.pins()) {
      REXLOG_INFO("input_player_slots: player {} is the pad matching \"{}\"", pin.user + 1,
                  pin.match);
    }
  }
  slot_table_.SetVirtualPolicy(
      ParseVirtualPadPolicy(REXCVAR_GET(input_virtual_pads)).value_or(VirtualPadPolicy::kAfterPhysical));
  slot_table_.SetRememberDisconnected(REXCVAR_GET(input_remember_slots));
}

void InputSystem::RefreshDevices() {
  ApplySlotConfig();

  std::vector<DeviceInfo> seen;
  std::vector<InputDriver*> owners;
  std::vector<DeviceInfo> enumerated;
  for (auto& driver : drivers_) {
    enumerated.clear();
    driver->EnumerateDevices(enumerated);
    for (auto& info : enumerated) {
      if (info.virtual_device && !info.synthetic &&
          slot_table_.virtual_policy() == VirtualPadPolicy::kIgnore) {
        if (ignored_virtual_.insert(info.id).second) {
          REXLOG_WARN("Input: ignoring virtual pad {} (input_virtual_pads = ignore).",
                      DescribeDevice(info));
        }
        continue;
      }
      seen.push_back(info);
      owners.push_back(driver.get());
    }
  }

  // Carry forward ordinals already handed out, so a device keeps its guest user
  // when another pad is unplugged.
  bool changed = seen.size() != devices_.size();
  std::vector<bool> fresh(seen.size(), false);
  for (size_t i = 0; i < seen.size(); i++) {
    auto existing = std::find_if(devices_.begin(), devices_.end(),
                                 [&](const DeviceInfo& d) { return d.id == seen[i].id; });
    if (existing != devices_.end()) {
      seen[i].ordinal = existing->ordinal;
      continue;
    }
    fresh[i] = true;
    changed = true;
  }

  // Departures first, so a pad that left and came back within one refresh
  // finds its reservation.
  for (const auto& old : devices_) {
    if (std::none_of(seen.begin(), seen.end(),
                     [&](const DeviceInfo& d) { return d.id == old.id; })) {
      active_devices_.Forget(old.id);
      changed = true;
      if (!old.synthetic) {
        slot_table_.OnDisconnected(old);
        REXLOG_INFO("Input: player {} disconnected: {}{}", old.ordinal + 1, DescribeDevice(old),
                    old.ordinal < kMaxGuestUsers && REXCVAR_GET(input_remember_slots)
                        ? "; the player is kept for it until another pad needs it"
                        : "");
      }
    }
  }

  // Runs after the carry-forward pass so a new device cannot take an ordinal a
  // live one is still holding. New pads are placed one at a time in the
  // drivers' connection order, each seeing the ones placed before it.
  std::vector<DeviceInfo> live;
  for (size_t i = 0; i < seen.size(); i++) {
    if (!fresh[i] || seen[i].synthetic) {
      live.push_back(seen[i]);
    }
  }
  for (size_t i = 0; i < seen.size(); i++) {
    if (!fresh[i]) {
      continue;
    }
    // Only physical devices consume an ordinal, so the first pad to connect is
    // guest user 0 however many synthetic devices enumerated ahead of it.
    if (seen[i].synthetic) {
      seen[i].ordinal = kSyntheticOrdinal;
      continue;
    }
    seen[i].ordinal = slot_table_.AssignOrdinal(seen[i], live);
    live.push_back(seen[i]);
    if (seen[i].ordinal < kMaxGuestUsers) {
      REXLOG_INFO("Input: player {} connected: {}", seen[i].ordinal + 1, DescribeDevice(seen[i]));
    } else {
      REXLOG_WARN("Input: {} connected, but all {} players already have a controller; it is not "
                  "used until one leaves and it is reconnected.",
                  DescribeDevice(seen[i]), kMaxGuestUsers);
    }
    if (IsSteamVirtualPad(seen[i].vendor_id, seen[i].product_id, seen[i].name) &&
        !steam_virtual_warned_) {
      steam_virtual_warned_ = true;
      REXLOG_WARN(
          "Input: {} is Steam Input's virtual pad, so Steam has taken a real controller. "
          "Disable Steam Input for this game or quit Steam to use the controller directly.",
          DescribeDevice(seen[i]));
    }
  }

  std::vector<size_t> order(seen.size());
  for (size_t i = 0; i < order.size(); i++) {
    order[i] = i;
  }
  // Stable: synthetic devices share one ordinal, and their relative order
  // decides which answers GetCapabilities when no pad is attached.
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return seen[a].ordinal < seen[b].ordinal; });

  devices_.clear();
  device_owners_.clear();
  for (size_t i : order) {
    devices_.push_back(seen[i]);
    device_owners_.push_back(owners[i]);
  }

  if (changed && assignment_) {
    assignment_->OnDevicesChanged(devices_);
    // Player lights follow the guest user, not the driver's own order.
    for (size_t i = 0; i < devices_.size(); i++) {
      if (!devices_[i].synthetic) {
        device_owners_[i]->OnDeviceUserChanged(devices_[i].id, UserForDevice(devices_[i].id));
      }
    }
  }
}

int32_t InputSystem::UserForDevice(DeviceId id) const {
  if (!assignment_) {
    return -1;
  }
  std::vector<DeviceId> ids;
  for (uint32_t user = 0; user < kMaxGuestUsers; user++) {
    assignment_->DevicesForUser(user, ids);
    if (std::find(ids.begin(), ids.end(), id) != ids.end()) {
      return static_cast<int32_t>(user);
    }
  }
  return -1;
}

std::vector<InputSystem::DeviceSnapshot> InputSystem::SnapshotDevices() {
  RefreshDevices();
  std::vector<DeviceSnapshot> out;
  out.reserve(devices_.size());
  for (size_t i = 0; i < devices_.size(); i++) {
    DeviceSnapshot snap;
    snap.info = devices_[i];
    snap.user = UserForDevice(devices_[i].id);
    snap.has_state = device_owners_[i]->GetDeviceState(devices_[i].id, &snap.state) ==
                     X_ERROR_SUCCESS;
    snap.has_caps = device_owners_[i]->GetDeviceCapabilities(devices_[i].id, 0, &snap.caps) ==
                    X_ERROR_SUCCESS;
    out.push_back(std::move(snap));
  }
  return out;
}

void InputSystem::LogStateIfDue() {
  if (!REXCVAR_GET(input_log_state)) {
    return;
  }
  const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  int64_t last = last_state_log_ms_.load();
  if (now - last < kStateLogInterval.count() ||
      !last_state_log_ms_.compare_exchange_strong(last, now)) {
    return;
  }
  const auto snapshots = SnapshotDevices();
  if (snapshots.empty()) {
    REXLOG_INFO("Input state: no controllers");
  }
  for (const auto& snap : snapshots) {
    const std::string player =
        snap.user >= 0 ? fmt::format("P{}", snap.user + 1) : std::string("--");
    REXLOG_INFO("Input state: {} {} {}", player, DescribeDevice(snap.info),
                snap.has_state ? FormatGamepad(snap.state.gamepad) : std::string("no state"));
  }
}

InputDriver* InputSystem::DriverForDevice(DeviceId id) {
  for (size_t i = 0; i < devices_.size(); i++) {
    if (devices_[i].id == id) {
      return device_owners_[i];
    }
  }
  return nullptr;
}

const DeviceInfo* InputSystem::DeviceInfoFor(DeviceId id) const {
  for (const auto& device : devices_) {
    if (device.id == id) {
      return &device;
    }
  }
  return nullptr;
}

DeviceId InputSystem::ChooseDeviceForUser(uint32_t user_index) const {
  if (!assignment_) {
    return DeviceId::kInvalid;
  }
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);
  if (ids.empty()) {
    return DeviceId::kInvalid;
  }
  // Prefer the pad in hand, so button glyphs follow it rather than whichever
  // device enumerated first.
  DeviceId chosen = active_devices_.Active(user_index);
  if (std::find(ids.begin(), ids.end(), chosen) == ids.end()) {
    chosen = ids.front();
  }
  return chosen;
}

X_RESULT InputSystem::GetCapabilities(uint32_t user_index, uint32_t flags,
                                      X_INPUT_CAPABILITIES* out_caps) {
  SCOPE_profile_cpu_f("hid");
  if (!out_caps || !assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  DeviceId chosen = ChooseDeviceForUser(user_index);
  auto* driver = DriverForDevice(chosen);
  if (!driver) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  return driver->GetDeviceCapabilities(chosen, flags, out_caps);
}

void InputSystem::UpdateConnectedUser(uint32_t user_index, bool connected) {
  if (user_index >= kMaxGuestUsers || connected_users_.test(user_index) == connected) {
    return;
  }
  connected_users_.set(user_index, connected);
  if (connected) {
    REXLOG_INFO("New controller connected to slot {}.", user_index);
  } else {
    REXLOG_INFO("Controller disconnected from slot {}.", user_index);
  }

  // Titles poll capabilities off this rather than every frame, so without it
  // a pad plugged in mid-game is never noticed.
  if (auto* kernel_state = REX_KERNEL_STATE()) {
    kernel_state->BroadcastNotification(kXNotificationSystemInputDevicesChanged, 0);
  }

  if (!connected) {
    user_max_joystick_value_[user_index] = {};
    consumed_buttons_[user_index] = 0;
    return;
  }

  // Deadzone percentages scale against the device's own range.
  DeviceId chosen = ChooseDeviceForUser(user_index);
  auto* driver = DriverForDevice(chosen);
  if (!driver) {
    return;
  }
  X_INPUT_CAPABILITIES caps = {};
  if (driver->GetDeviceCapabilities(chosen, 0, &caps) != X_ERROR_SUCCESS) {
    return;
  }
  user_max_joystick_value_[user_index] = {{caps.gamepad.thumb_lx, caps.gamepad.thumb_ly},
                                          {caps.gamepad.thumb_rx, caps.gamepad.thumb_ry}};
}

X_RESULT InputSystem::GetState(uint32_t user_index, X_INPUT_STATE* out_state) {
  SCOPE_profile_cpu_f("hid");

  // A dialog owns the controller.
  if (ui_input_blockers_.load() > 0) {
    if (out_state) {
      std::memset(out_state, 0, sizeof(*out_state));
    }
    return X_ERROR_SUCCESS;
  }

  X_RESULT result = GetStateForUI(user_index, out_state);

  if (result == X_ERROR_SUCCESS && out_state && user_index < kMaxGuestUsers &&
      consumed_buttons_[user_index] != 0) {
    const uint16_t buttons = out_state->gamepad.buttons;
    // Each button leaves the mask once the player lets go of it.
    consumed_buttons_[user_index] &= buttons;
    out_state->gamepad.buttons = static_cast<uint16_t>(buttons & ~consumed_buttons_[user_index]);
  }

  return result;
}

X_RESULT InputSystem::GetStateForUI(uint32_t user_index, X_INPUT_STATE* out_state) {
  LogStateIfDue();
  SCOPE_profile_cpu_f("hid");
  if (!assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  X_INPUT_STATE merged = {};
  bool any = false;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    if (!driver) {
      continue;
    }
    X_INPUT_STATE state = {};
    if (driver->GetDeviceState(id, &state) != X_ERROR_SUCCESS) {
      continue;
    }
    active_devices_.Observe(user_index, id, state.gamepad);
    if (!any) {
      merged = state;
      any = true;
    } else {
      MergeInto(merged, state);
    }
  }

  if (!any) {
    UpdateConnectedUser(user_index, false);
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  UpdateConnectedUser(user_index, true);
  AdjustDeadzoneLevels(user_index, &merged.gamepad);
  if (static_cast<uint16_t>(merged.gamepad.buttons) != 0 && user_index < kMaxGuestUsers) {
    last_used_user_ = user_index;
  }
  if (out_state) {
    *out_state = merged;
  }
  return X_ERROR_SUCCESS;
}

void InputSystem::AddUIInputBlocker() {
  ui_input_blockers_.fetch_add(1);
}

void InputSystem::RemoveUIInputBlocker() {
  // Whatever is held right now stays masked until released, so the press that
  // dismissed the dialog is not also read as a press in the game.
  X_INPUT_STATE state = {};
  for (uint32_t user_index = 0; user_index < kMaxGuestUsers; user_index++) {
    if (GetStateForUI(user_index, &state) == X_ERROR_SUCCESS) {
      consumed_buttons_[user_index] |= static_cast<uint16_t>(state.gamepad.buttons);
    }
  }

  ui_input_blockers_.fetch_sub(1);
}

bool InputSystem::GetVibrationEnabled() const {
  return REXCVAR_GET(vibration);
}

void InputSystem::ToggleVibration() {
  REXCVAR_SET(vibration, !REXCVAR_GET(vibration));
  // The guest's next SetState may never come while a motor is running.
  X_INPUT_VIBRATION silence = {};
  for (uint32_t user_index = 0; user_index < kMaxGuestUsers; user_index++) {
    SetState(user_index, &silence);
  }
}

void InputSystem::AdjustDeadzoneLevels(uint32_t user_index, X_INPUT_GAMEPAD* gamepad) const {
  if (user_index >= kMaxGuestUsers || !gamepad) {
    return;
  }

  // Radial: the cutoff follows the stick's angle, so a diagonal is not held to
  // a larger displacement than a cardinal push.
  auto apply = [](double percentage, const JoystickValue& max, int16_t& axis_x, int16_t& axis_y) {
    if (percentage <= 0.0 || percentage >= 1.0) {
      return;
    }
    const double deadzone_x = max.first * percentage;
    const double deadzone_y = max.second * percentage;
    const double theta = std::atan2(static_cast<double>(axis_y), static_cast<double>(axis_x));
    const double cutoff_x = std::cos(theta) * deadzone_x;
    const double cutoff_y = std::sin(theta) * deadzone_y;
    if (axis_x > -cutoff_x && axis_x < cutoff_x) {
      axis_x = 0;
    }
    if (axis_y > -cutoff_y && axis_y < cutoff_y) {
      axis_y = 0;
    }
  };

  const auto& maxima = user_max_joystick_value_[user_index];
  int16_t lx = gamepad->thumb_lx;
  int16_t ly = gamepad->thumb_ly;
  apply(REXCVAR_GET(left_stick_deadzone_percentage), maxima.first, lx, ly);
  gamepad->thumb_lx = lx;
  gamepad->thumb_ly = ly;

  int16_t rx = gamepad->thumb_rx;
  int16_t ry = gamepad->thumb_ry;
  apply(REXCVAR_GET(right_stick_deadzone_percentage), maxima.second, rx, ry);
  gamepad->thumb_rx = rx;
  gamepad->thumb_ry = ry;
}

X_INPUT_VIBRATION InputSystem::ModifyVibrationLevel(const X_INPUT_VIBRATION* vibration) const {
  X_INPUT_VIBRATION modified = *vibration;
  if (!REXCVAR_GET(vibration)) {
    modified.left_motor_speed = 0;
    modified.right_motor_speed = 0;
  }
  return modified;
}

X_RESULT InputSystem::SetState(uint32_t user_index, X_INPUT_VIBRATION* vibration) {
  SCOPE_profile_cpu_f("hid");
  if (!assignment_ || !vibration) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  const X_INPUT_VIBRATION modified = ModifyVibrationLevel(vibration);

  // Every pad on this user belongs to the same player, so all of them buzz.
  // Only pads decide the result: synthetic devices accept any vibration and
  // would otherwise report success for a pad that never rumbled.
  bool any_pad = false;
  bool any_synthetic = false;
  bool pad_rumbled = false;
  X_RESULT pad_error = X_ERROR_DEVICE_NOT_CONNECTED;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    const DeviceInfo* info = DeviceInfoFor(id);
    if (!driver || !info) {
      continue;
    }
    X_INPUT_VIBRATION per_device = modified;
    X_RESULT result = driver->SetDeviceVibration(id, &per_device);
    if (info->synthetic) {
      any_synthetic = true;
      continue;
    }
    any_pad = true;
    if (result == X_ERROR_SUCCESS) {
      pad_rumbled = true;
    } else {
      pad_error = result;
    }
  }
  if (any_pad) {
    return pad_rumbled ? X_ERROR_SUCCESS : pad_error;
  }
  return any_synthetic ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT InputSystem::GetKeystroke(uint32_t user_index, uint32_t flags,
                                   X_INPUT_KEYSTROKE* out_keystroke) {
  SCOPE_profile_cpu_f("hid");
  if (!assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  bool any_connected = false;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    if (!driver) {
      continue;
    }
    X_RESULT result = driver->GetDeviceKeystroke(id, flags, out_keystroke);
    if (result == X_ERROR_SUCCESS) {
      out_keystroke->user_index = static_cast<uint8_t>(user_index);
      return result;
    }
    if (result != X_ERROR_DEVICE_NOT_CONNECTED) {
      any_connected = true;
    }
  }
  return any_connected ? X_ERROR_EMPTY : X_ERROR_DEVICE_NOT_CONNECTED;
}

std::unique_ptr<InputDriver> CreateScriptPadDriverFromConfig() {
  script::ScriptFile file;
  const std::string& path = REXCVAR_GET(hid_virtual_pad_script);
  if (!path.empty()) {
    std::ifstream stream(path);
    if (!stream) {
      REXLOG_ERROR("hid_virtual_pad_script: cannot read '{}'", path);
    } else {
      std::stringstream text;
      text << stream.rdbuf();
      file = script::ParseScriptFile(text.str());
    }
  }
  const std::array<std::string, kMaxGuestUsers> inline_scripts = {
      REXCVAR_GET(hid_virtual_pad1), REXCVAR_GET(hid_virtual_pad2),
      REXCVAR_GET(hid_virtual_pad3), REXCVAR_GET(hid_virtual_pad4)};
  int count = REXCVAR_GET(hid_virtual_pads);
  if (count == 0) {
    count = std::clamp(file.pad_count, 0, static_cast<int>(kMaxGuestUsers));
  }
  if (count == 0) {
    return nullptr;
  }
  const int look = file.look_speed > 0 ? file.look_speed : REXCVAR_GET(hid_virtual_pad_look_speed);
  std::vector<std::vector<script::Step>> scripts;
  for (int i = 0; i < count; i++) {
    const std::string& text = inline_scripts[i].empty() ? file.pads[i] : inline_scripts[i];
    std::string error;
    scripts.push_back(script::ParseScript(text, &error));
    if (!error.empty()) {
      REXLOG_WARN("Virtual pad {}: {}", i + 1, error);
    }
    REXLOG_INFO("Virtual pad {} on player {}: {} step(s){}", i + 1, i + 1, scripts.back().size(),
                scripts.back().empty() ? ", idle" : "");
  }
  return std::make_unique<script::ScriptInputDriver>(std::move(scripts), look);
}

std::unique_ptr<InputSystem> CreateDefaultInputSystem(bool tool_mode) {
  auto input = std::make_unique<InputSystem>(nullptr);

  if (!tool_mode) {
    // First, so the script pads are on their players before any controller.
    if (auto script_driver = CreateScriptPadDriverFromConfig()) {
      input->AddDriver(std::move(script_driver));
    }

    const std::string steam_warning = SteamInputWarning(DetectSteamInput());
    if (!steam_warning.empty()) {
      REXLOG_WARN("Input: {}", steam_warning);
    }
#if REX_PLATFORM_WIN32
    if (REXCVAR_GET(input_backend) == "xinput") {
      auto xinput_driver = std::make_unique<xinput::XinputInputDriver>(nullptr, 0);
      if (xinput_driver->Setup() == X_STATUS_SUCCESS) {
        input->AddDriver(std::move(xinput_driver));
      }
    }
#endif

    if (REXCVAR_GET(input_backend) == "sdl") {
      auto sdl_driver = std::make_unique<sdl::SDLInputDriver>(nullptr, 0);
      if (sdl_driver->Setup() == X_STATUS_SUCCESS) {
        input->AddDriver(std::move(sdl_driver));
      }
    }

    // MnK driver (keyboard/mouse -> controller emulation)
    auto mnk_driver = std::make_unique<mnk::MnkInputDriver>(nullptr, 0);
    if (mnk_driver->Setup() == X_STATUS_SUCCESS) {
      input->AddDriver(std::move(mnk_driver));
    }
  }

  // NOP driver (primary in tool mode, fallback otherwise)
  uint8_t nop_index = tool_mode ? 0 : 1;
  input->AddDriver(std::make_unique<nop::NopInputDriver>(nullptr, nop_index));
  input->SetDeviceAssignment(std::make_unique<SlotAssignment>());
  return input;
}

}  // namespace rex::input
