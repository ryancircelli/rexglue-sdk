#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <rex/input/device_assignment.h>
#include <rex/input/input_system.h>
#include <rex/input/slot_table.h>
#include <rex/input/state_log.h>
#include <rex/input/steam_detect.h>

using rex::X_RESULT;
using rex::X_STATUS;

using rex::input::DeviceId;
using rex::input::DeviceInfo;
using rex::input::InputDriver;
using rex::input::InputSystem;
using rex::input::kMaxGuestUsers;
using rex::input::SlotAssignment;
using rex::input::SlotTable;
using rex::input::VirtualPadPolicy;
using rex::input::X_INPUT_CAPABILITIES;
using rex::input::X_INPUT_KEYSTROKE;
using rex::input::X_INPUT_STATE;
using rex::input::X_INPUT_VIBRATION;

namespace {

DeviceInfo Pad(uint64_t id, std::string serial, std::string name = "Xbox Wireless Controller") {
  DeviceInfo info;
  info.id = static_cast<DeviceId>(id);
  info.serial = std::move(serial);
  info.name = std::move(name);
  info.guid = "050000005e040000e002000000000000";
  info.vendor_id = 0x045e;
  info.product_id = 0x0b13;
  return info;
}

DeviceInfo Virtual(uint64_t id) {
  DeviceInfo info;
  info.id = static_cast<DeviceId>(id);
  info.name = "Steam Virtual Gamepad";
  info.guid = "03000000de2800ff1100000000000000";
  info.vendor_id = 0x28de;
  info.product_id = 0x11ff;
  info.virtual_device = true;
  return info;
}

DeviceInfo Keyboard(int32_t fixed_user) {
  DeviceInfo info;
  info.id = static_cast<DeviceId>(0x4D4E4B00);
  info.name = "Keyboard and Mouse";
  info.synthetic = true;
  info.fixed_user = fixed_user;
  return info;
}

/// Mirrors InputSystem's use: connected pads with their ordinals.
struct Table {
  SlotTable table;
  std::vector<DeviceInfo> live;

  uint32_t Connect(DeviceInfo info) {
    info.ordinal = table.AssignOrdinal(info, live);
    live.push_back(info);
    return info.ordinal;
  }
  void Disconnect(uint64_t id) {
    for (auto it = live.begin(); it != live.end(); ++it) {
      if (it->id == static_cast<DeviceId>(id)) {
        table.OnDisconnected(*it);
        live.erase(it);
        return;
      }
    }
  }
};

}  // namespace

TEST_CASE("Pads take players in connection order", "[input][slots]") {
  Table t;
  CHECK(t.Connect(Pad(1, "A")) == 0);
  CHECK(t.Connect(Pad(2, "B")) == 1);
  CHECK(t.Connect(Pad(3, "C")) == 2);
  CHECK(t.Connect(Pad(4, "D")) == 3);
  // A fifth pad is connected but not given to the game.
  CHECK(t.Connect(Pad(5, "E")) >= kMaxGuestUsers);
}

TEST_CASE("A pad that reconnects gets its player back", "[input][slots]") {
  Table t;
  t.Connect(Pad(1, "A"));
  t.Connect(Pad(2, "B"));
  t.Connect(Pad(3, "C"));
  t.Disconnect(2);
  // A different pad arriving meanwhile does not take player 2.
  CHECK(t.Connect(Pad(4, "D")) == 3);
  // B comes back with a new device id and finds player 2 waiting.
  CHECK(t.Connect(Pad(5, "B")) == 1);
}

TEST_CASE("Reconnect order does not reshuffle players", "[input][slots]") {
  Table t;
  t.Connect(Pad(1, "A"));
  t.Connect(Pad(2, "B"));
  t.Connect(Pad(3, "C"));
  t.Disconnect(1);
  t.Disconnect(3);
  // They come back in the opposite order.
  CHECK(t.Connect(Pad(4, "C")) == 2);
  CHECK(t.Connect(Pad(5, "A")) == 0);
}

TEST_CASE("A waiting player goes to a new pad only when nothing else is free", "[input][slots]") {
  Table t;
  for (uint64_t i = 1; i <= 4; i++) {
    t.Connect(Pad(i, std::string(1, static_cast<char>('A' + i - 1))));
  }
  t.Disconnect(2);  // B leaves player 2
  CHECK(t.Connect(Pad(9, "Z")) == 1);
  // Z holds it now; B is not owed it any more.
  t.Disconnect(9);
  CHECK(t.Connect(Pad(10, "B")) == 1);
}

TEST_CASE("Identical pads without serials keep separate players", "[input][slots]") {
  Table t;
  t.Connect(Pad(1, ""));
  t.Connect(Pad(2, ""));
  t.Disconnect(1);
  t.Disconnect(2);
  CHECK(t.Connect(Pad(3, "")) == 0);
  CHECK(t.Connect(Pad(4, "")) == 1);
}

TEST_CASE("Virtual pads never take player 1 by default", "[input][slots]") {
  Table t;
  CHECK(t.Connect(Virtual(1)) == 1);
  CHECK(t.Connect(Pad(2, "A")) == 0);

  Table normal;
  normal.table.SetVirtualPolicy(VirtualPadPolicy::kNormal);
  CHECK(normal.Connect(Virtual(1)) == 0);
}

TEST_CASE("Config pins pads to players", "[input][slots]") {
  Table t;
  std::string error;
  t.table.SetPins(rex::input::ParseSlotPins("1=C; 3=045e:0b13 , 2=bogus=x;9=A;x", &error));
  CHECK_FALSE(error.empty());  // "9=A" and "x"
  REQUIRE(t.table.pins().size() == 3);

  // Names that no pin text appears in, so only serials and vid:pid match.
  // A is unpinned by serial but matches player 3's vid:pid pin.
  CHECK(t.Connect(Pad(1, "A", "Pad")) == 2);
  // B also matches the vid:pid pin, but player 3 is taken; player 1 is pinned
  // to C, so B takes player 2... which is pinned to "bogus=x", a pad that never
  // shows up, so B ends on player 4.
  CHECK(t.Connect(Pad(2, "B", "Pad")) == 3);
  CHECK(t.Connect(Pad(3, "C", "Pad")) == 0);
  // Nothing unpinned is free: the next pad takes a pinned player.
  CHECK(t.Connect(Pad(4, "D", "Pad")) == 1);
}

TEST_CASE("Pins match serial, GUID, vid:pid or name", "[input][slots]") {
  const DeviceInfo pad = Pad(1, "3032363030303139");
  CHECK(rex::input::PinMatches("3032363030303139", pad));
  CHECK(rex::input::PinMatches("050000005E040000E002000000000000", pad));
  CHECK(rex::input::PinMatches("045E:0B13", pad));
  CHECK(rex::input::PinMatches("wireless", pad));
  CHECK_FALSE(rex::input::PinMatches("dualsense", pad));
  CHECK_FALSE(rex::input::PinMatches("", pad));
}

TEST_CASE("Keyboard as its own player keeps that player", "[input][slots]") {
  Table t;
  t.live.push_back(Keyboard(1));
  CHECK(t.Connect(Pad(1, "A")) == 0);
  CHECK(t.Connect(Pad(2, "B")) == 2);

  SlotAssignment assignment;
  std::vector<DeviceInfo> devices = t.live;
  devices[0].ordinal = UINT32_MAX;
  assignment.OnDevicesChanged(devices);
  std::vector<DeviceId> ids;
  assignment.DevicesForUser(1, ids);
  REQUIRE(ids.size() == 1);
  CHECK(ids[0] == static_cast<DeviceId>(0x4D4E4B00));
  assignment.DevicesForUser(0, ids);
  CHECK(ids == std::vector<DeviceId>{static_cast<DeviceId>(1)});
}

TEST_CASE("Steam virtual pad detection", "[input][steam]") {
  CHECK(rex::input::IsSteamVirtualPad(0x28de, 0x11ff, ""));
  CHECK(rex::input::IsSteamVirtualPad(0, 0, "Steam Virtual Gamepad"));
  CHECK_FALSE(rex::input::IsSteamVirtualPad(0x28de, 0x1205, "Steam Deck"));
  CHECK_FALSE(rex::input::IsSteamVirtualPad(0x045e, 0x0b13, "Xbox Wireless Controller"));

  rex::input::SteamInputReport report;
  CHECK(rex::input::SteamInputWarning(report).empty());
  report.steam_running = true;
  report.settings = {"Xbox=1"};
  report.steam_input_enabled = true;
  const std::string warning = rex::input::SteamInputWarning(report);
  CHECK(warning.find("Steam is running") != std::string::npos);
  CHECK(warning.find("Xbox=1") != std::string::npos);
}

TEST_CASE("Pad state formats readably", "[input][log]") {
  rex::input::X_INPUT_GAMEPAD pad = {};
  CHECK(rex::input::ButtonNames(0) == "-");
  pad.buttons = rex::input::X_INPUT_GAMEPAD_A | rex::input::X_INPUT_GAMEPAD_START;
  pad.right_trigger = 255;
  pad.thumb_ly = static_cast<uint16_t>(static_cast<int16_t>(-32768));
  CHECK(rex::input::FormatGamepad(pad) == "buttons=[A Start] LT=0 RT=255 L=(0,-32768) R=(0,0)");
}

namespace {

/// Pads with identities, plugged and unplugged between polls.
class IdentityDriver final : public InputDriver {
 public:
  IdentityDriver() : InputDriver(nullptr, 0) {}
  X_STATUS Setup() override { return X_STATUS_SUCCESS; }
  void Plug(DeviceInfo info) { devices_.push_back(std::move(info)); }
  void Unplug(uint64_t id) {
    std::erase_if(devices_, [&](const DeviceInfo& d) { return d.id == static_cast<DeviceId>(id); });
  }
  void EnumerateDevices(std::vector<DeviceInfo>& out) override {
    out.insert(out.end(), devices_.begin(), devices_.end());
  }
  X_RESULT GetDeviceState(DeviceId id, X_INPUT_STATE* out_state) override {
    for (const auto& d : devices_) {
      if (d.id == id) {
        *out_state = {};
        return X_ERROR_SUCCESS;
      }
    }
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  X_RESULT GetDeviceCapabilities(DeviceId, uint32_t, X_INPUT_CAPABILITIES* out_caps) override {
    *out_caps = {};
    return X_ERROR_SUCCESS;
  }
  X_RESULT SetDeviceVibration(DeviceId, X_INPUT_VIBRATION*) override { return X_ERROR_SUCCESS; }
  X_RESULT GetDeviceKeystroke(DeviceId, uint32_t, X_INPUT_KEYSTROKE*) override {
    return X_ERROR_EMPTY;
  }
  void OnDeviceUserChanged(DeviceId id, int32_t user) override { lights[id] = user; }

  std::map<DeviceId, int32_t> lights;

 private:
  std::vector<DeviceInfo> devices_;
};

std::map<uint64_t, int32_t> Players(InputSystem& input) {
  std::map<uint64_t, int32_t> out;
  for (const auto& snap : input.SnapshotDevices()) {
    out[static_cast<uint64_t>(snap.info.id)] = snap.user;
  }
  return out;
}

}  // namespace

TEST_CASE("Hot-plug mid-game keeps every other player in place", "[input][slots]") {
  InputSystem input(nullptr);
  auto driver_owner = std::make_unique<IdentityDriver>();
  IdentityDriver* driver = driver_owner.get();
  input.AddDriver(std::move(driver_owner));
  input.SetDeviceAssignment(std::make_unique<SlotAssignment>());

  driver->Plug(Pad(1, "A"));
  driver->Plug(Pad(2, "B"));
  driver->Plug(Pad(3, "C"));
  driver->Plug(Pad(4, "D"));
  CHECK(Players(input) == std::map<uint64_t, int32_t>{{1, 0}, {2, 1}, {3, 2}, {4, 3}});
  CHECK(driver->lights[static_cast<DeviceId>(3)] == 2);

  // Player 2's battery dies; players 1, 3 and 4 do not move.
  driver->Unplug(2);
  CHECK(Players(input) == std::map<uint64_t, int32_t>{{1, 0}, {3, 2}, {4, 3}});

  // It comes back over Bluetooth with a new device id: player 2 again.
  driver->Plug(Pad(7, "B"));
  CHECK(Players(input) == std::map<uint64_t, int32_t>{{1, 0}, {3, 2}, {4, 3}, {7, 1}});
  CHECK(driver->lights[static_cast<DeviceId>(7)] == 1);
}

TEST_CASE("Steam's virtual pad does not push a real pad off player 1", "[input][slots]") {
  InputSystem input(nullptr);
  auto driver_owner = std::make_unique<IdentityDriver>();
  IdentityDriver* driver = driver_owner.get();
  input.AddDriver(std::move(driver_owner));
  input.SetDeviceAssignment(std::make_unique<SlotAssignment>());

  driver->Plug(Virtual(1));
  driver->Plug(Pad(2, "A"));
  CHECK(Players(input) == std::map<uint64_t, int32_t>{{1, 1}, {2, 0}});
}

#include <rex/input/script/script_input_driver.h>

namespace sc = rex::input::script;

TEST_CASE("Pad scripts parse taps, holds, combos and disconnects", "[input][script]") {
  std::string error;
  const auto steps =
      sc::ParseScript("25:a, 60-90:stick_up+look_right;100-105:disconnect 7:bogus 8:disconnect",
                      &error);
  REQUIRE(steps.size() == 3);
  CHECK(steps[0].start == 25);
  CHECK(steps[0].end == 25.25);
  CHECK(steps[0].controls == rex::input::X_INPUT_GAMEPAD_A);
  CHECK(steps[1].controls == (sc::kControlMoveUp | sc::kControlLookRight));
  CHECK(steps[2].disconnect);
  CHECK(error.find("7:bogus") != std::string::npos);
  CHECK(error.find("8:disconnect") != std::string::npos);

  CHECK(sc::Evaluate(steps, 25.1).controls == rex::input::X_INPUT_GAMEPAD_A);
  CHECK(sc::Evaluate(steps, 25.3).controls == 0);
  CHECK_FALSE(sc::Evaluate(steps, 101).connected);
  CHECK(sc::Evaluate(steps, 105).connected);
}

TEST_CASE("Script files take pad keys and bo2 route keys", "[input][script]") {
  const auto file = sc::ParseScriptFile(
      "# a route\nduration_s = 300\nplayers = 4\nBO2ZM_TEST_BUTTONS = 25:a\n"
      "BO2ZM_TEST_BUTTONS_P3 = 30:b\npad2 = 40:x\nlook_speed = 50\n");
  CHECK(file.pad_count == 4);
  CHECK(file.look_speed == 50);
  CHECK(file.pads[0] == "25:a");
  CHECK(file.pads[1] == "40:x");
  CHECK(file.pads[2] == "30:b");
  CHECK(file.pads[3].empty());
}

TEST_CASE("Script controls map to sticks and triggers", "[input][script]") {
  rex::input::X_INPUT_GAMEPAD pad = {};
  sc::ControlsToGamepad(sc::kControlMoveUp | sc::kControlLookLeft | sc::kControlRT, 50, &pad);
  CHECK(static_cast<int16_t>(pad.thumb_ly) == 30000);
  CHECK(static_cast<int16_t>(pad.thumb_rx) == -16383);
  CHECK(pad.right_trigger == 255);
  CHECK(pad.left_trigger == 0);
}

TEST_CASE("Four script pads take players 1-4 and survive a scripted unplug", "[input][script]") {
  double now = 0;
  std::vector<std::vector<sc::Step>> scripts = {
      sc::ParseScript("1-2:a"), sc::ParseScript("5-6:disconnect"), {}, sc::ParseScript("1-3:stick_up")};
  InputSystem input(nullptr);
  auto real_owner = std::make_unique<IdentityDriver>();
  IdentityDriver* real = real_owner.get();
  input.AddDriver(std::make_unique<sc::ScriptInputDriver>(std::move(scripts), 35,
                                                         [&now] { return now; }));
  input.AddDriver(std::move(real_owner));
  input.SetDeviceAssignment(std::make_unique<SlotAssignment>());

  const uint64_t base = sc::ScriptInputDriver::kScriptPadBase;
  CHECK(Players(input) ==
        std::map<uint64_t, int32_t>{{base, 0}, {base + 1, 1}, {base + 2, 2}, {base + 3, 3}});

  now = 1.5;
  X_INPUT_STATE state = {};
  REQUIRE(input.GetState(0, &state) == X_ERROR_SUCCESS);
  CHECK(state.gamepad.buttons == rex::input::X_INPUT_GAMEPAD_A);
  REQUIRE(input.GetState(3, &state) == X_ERROR_SUCCESS);
  CHECK(static_cast<int16_t>(state.gamepad.thumb_ly) > 0);

  // Pad 2 unplugged: a real controller arriving now goes past the players held.
  now = 5.5;
  real->Plug(Pad(10, "REAL"));
  auto players = Players(input);
  CHECK(players.count(base + 1) == 0);
  CHECK(players[10] == 1);  // the only free player

  // The real pad leaves, pad 2 returns to its player.
  real->Unplug(10);
  now = 7;
  players = Players(input);
  CHECK(players[base + 1] == 1);
}
