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

#include <rex/input/script/script_events.h>
#include <rex/logging.h>

TEST_CASE("Pad scripts parse wait steps", "[input][script]") {
  std::string error;
  const auto steps = sc::ParseScript(
      "1:a, wait_event lobby_open 40, 2:start; wait_log Local lobby open, 30:wait_log \"in a "
      "match, really\" 90\n0-1:b wait_event two words, wait_bogus x",
      &error);
  REQUIRE(steps.size() == 6);
  CHECK(steps[0].segment == 0);
  CHECK(steps[1].wait == sc::Step::Wait::kEvent);
  CHECK(steps[1].target == "lobby_open");
  CHECK(steps[1].timeout == 40);
  CHECK(steps[2].segment == 1);
  CHECK(steps[3].wait == sc::Step::Wait::kLog);
  CHECK(steps[3].target == "Local lobby open");
  CHECK(steps[3].timeout == sc::kDefaultWaitTimeout);
  CHECK(steps[4].target == "in a match, really");
  CHECK(steps[4].start == 30);
  CHECK(steps[4].timeout == 90);
  CHECK(steps[5].segment == 3);
  CHECK(steps[5].controls == rex::input::X_INPUT_GAMEPAD_B);
  // A wait has to start its own piece; here it is read as bad timed steps.
  CHECK(error.find("\"wait_event\"") != std::string::npos);
  CHECK(error.find("wait_bogus") != std::string::npos);
}

namespace {

uint16_t Buttons(sc::ScriptInputDriver& driver, uint32_t pad = 0) {
  X_INPUT_STATE state = {};
  if (driver.GetDeviceState(static_cast<rex::input::DeviceId>(
                                sc::ScriptInputDriver::kScriptPadBase + pad),
                            &state) != X_ERROR_SUCCESS) {
    return 0xFFFF;
  }
  return state.gamepad.buttons;
}

}  // namespace

TEST_CASE("A script waits for an event, then times later steps from it", "[input][script]") {
  sc::ResetScriptEvents();
  double now = 0;
  sc::ScriptInputDriver driver(
      {sc::ParseScript("0-1:a, wait_event lobby_open 60, 2-3:start"),
       sc::ParseScript("wait_event lobby_open, 1-2:x")},
      35, [&now] { return now; });
  CHECK(Buttons(driver) == rex::input::X_INPUT_GAMEPAD_A);
  now = 2.5;
  CHECK(Buttons(driver) == 0);  // waiting
  now = 30;
  CHECK(Buttons(driver) == 0);
  CHECK(Buttons(driver, 1) == 0);
  rex::input::SignalScriptEvent("lobby_open");
  now = 31;
  CHECK(Buttons(driver) == 0);  // 1 s after the event: nothing yet
  CHECK(Buttons(driver, 1) == 0);
  now = 32.5;  // pad 2 has waited since 0 and saw the event at 31
  CHECK(Buttons(driver, 1) == rex::input::X_INPUT_GAMEPAD_X);
  CHECK(Buttons(driver) == 0);
  now = 33.5;
  CHECK(Buttons(driver) == rex::input::X_INPUT_GAMEPAD_START);
}

TEST_CASE("Each pad takes its own copy of a signal, sent early or late", "[input][script]") {
  sc::ResetScriptEvents();
  double now = 0;
  rex::input::SignalScriptEvent("go");  // before anyone waits
  sc::ScriptInputDriver driver(
      {sc::ParseScript("wait_event go, 0-1:a"), sc::ParseScript("wait_event go, 0-1:b")}, 35,
      [&now] { return now; });
  now = 0.5;  // both waits end at once, and their steps start from here
  CHECK(Buttons(driver) == rex::input::X_INPUT_GAMEPAD_A);
  CHECK(Buttons(driver, 1) == rex::input::X_INPUT_GAMEPAD_B);
  now = 1.6;
  CHECK(Buttons(driver) == 0);
  // The one signal is used up: a second wait would block.
  CHECK(sc::ScriptEventCount("go") == 1);
}

TEST_CASE("A wait gives up after its timeout", "[input][script]") {
  sc::ResetScriptEvents();
  double now = 0;
  sc::ScriptInputDriver driver({sc::ParseScript("wait_event never 5, 1-2:y")}, 35,
                               [&now] { return now; });
  now = 4;
  CHECK(Buttons(driver) == 0);
  now = 5;
  CHECK(Buttons(driver) == 0);  // gave up here
  now = 6.5;
  CHECK(Buttons(driver) == rex::input::X_INPUT_GAMEPAD_Y);
}

TEST_CASE("wait_log goes on when a log line contains the text", "[input][script]") {
  sc::ResetScriptEvents();
  double now = 0;
  sc::ScriptInputDriver driver({sc::ParseScript("wait_log \"Local lobby open\" 60, 0-1:a")}, 35,
                               [&now] { return now; });
  now = 1;
  CHECK(Buttons(driver) == 0);
  sc::OnLogLine("[controllers] Local lobby closed");
  now = 2;
  CHECK(Buttons(driver) == 0);
  sc::OnLogLine("[controllers] Local lobby open");
  now = 3;  // seen now; A goes down at once
  CHECK(Buttons(driver) == rex::input::X_INPUT_GAMEPAD_A);
  now = 4.5;
  CHECK(Buttons(driver) == 0);
}

TEST_CASE("wait_log hears lines through the SDK logger", "[input][script]") {
  sc::ResetScriptEvents();
  sc::WatchLogText("marker for the wait test");
  REXLOG_INFO("a marker for the wait test, logged");
  CHECK(sc::ScriptEventCount(sc::LogEventName("marker for the wait test")) == 1);
}

// Four identical wired Xbox Series pads on macOS: SDL reports no serial and the
// same GUID for each (SDL issues #11742, #16566).

namespace {

DeviceInfo SeriesPad(uint64_t id, std::string location = "", std::string path = "") {
  DeviceInfo info;
  info.id = static_cast<DeviceId>(id);
  info.name = "Controller";
  info.guid = "030000005e040000120b00000000006d";  // ends in 'm': GameController pad
  info.vendor_id = 0x045e;
  info.product_id = 0x0b12;
  info.transport = "usb";
  info.location = std::move(location);
  info.path = std::move(path);
  return info;
}

}  // namespace

TEST_CASE("Identity chain: serial, then port, then path, then GUID", "[input][slots][ids]") {
  using rex::input::IdentitySource;
  using rex::input::ResolveDeviceIdentity;

  // A serial wins even when the port is known, so serial pads behave as before.
  DeviceInfo with_serial = SeriesPad(1, "14200000");
  with_serial.serial = "3032363030303139";
  CHECK(ResolveDeviceIdentity(with_serial).key == "serial:3032363030303139");
  CHECK(ResolveDeviceIdentity(with_serial).source == IdentitySource::kSerial);

  const auto port = ResolveDeviceIdentity(SeriesPad(2, "14200000"));
  CHECK(port.key == "port:045e:0b12@14200000");
  CHECK(port.source == IdentitySource::kLocation);
  CHECK(port.unique);

  // macOS HIDAPI paths and Linux event nodes change on every replug.
  for (const char* path :
       {"DevSrvsID:4294969871", "/dev/input/event7", "/dev/hidraw2", "XInput#1"}) {
    const auto id = ResolveDeviceIdentity(SeriesPad(3, "", path));
    CHECK(id.source == IdentitySource::kGuid);
    CHECK(id.key == "guid:030000005e040000120b00000000006d");
    CHECK_FALSE(id.unique);
    CHECK(id.reason.find("connection order") != std::string::npos);
  }
  // A Windows HID path names the port.
  const auto win = ResolveDeviceIdentity(
      SeriesPad(4, "", "\\\\?\\HID#VID_045E&PID_0B12&IG_00#8&1a2b3c4d&0&0000#{4d1e55b2}"));
  CHECK(win.source == IdentitySource::kPath);
  CHECK(win.unique);

  DeviceInfo unlocated = SeriesPad(5);
  unlocated.location_note = "no free player index to tag the GameController pad with";
  CHECK(ResolveDeviceIdentity(unlocated).reason.find("no free player index") != std::string::npos);
}

TEST_CASE("Without ports, four identical pads share one key", "[input][slots][ids]") {
  // The fallback that existed before ports: GUID plus connection order. One pad
  // replugged gets its player back; two replugged in swapped order swap.
  Table t;
  for (uint64_t i = 1; i <= 4; i++) {
    CHECK(t.Connect(SeriesPad(i)) == i - 1);
  }
  CHECK(rex::input::DeviceIdentityKey(SeriesPad(1)) == rex::input::DeviceIdentityKey(SeriesPad(4)));
  t.Disconnect(3);
  CHECK(t.Connect(SeriesPad(13)) == 2);

  t.Disconnect(2);  // player 2 leaves first
  t.Disconnect(4);  // then player 4
  // The pad that was player 4 is plugged back first: it cannot be told apart
  // and takes the oldest waiting player.
  CHECK(t.Connect(SeriesPad(14)) == 1);
  CHECK(t.Connect(SeriesPad(12)) == 3);
}

TEST_CASE("Four identical pads known by port keep their players", "[input][slots][ids]") {
  const char* ports[] = {"14100000", "14200000", "14300000", "14400000"};
  Table t;
  for (uint64_t i = 1; i <= 4; i++) {
    CHECK(t.Connect(SeriesPad(i, ports[i - 1])) == i - 1);
  }
  CHECK(rex::input::DeviceIdentityKey(t.live[0]) != rex::input::DeviceIdentityKey(t.live[1]));

  // Hot-plug of one.
  t.Disconnect(3);
  CHECK(t.Connect(SeriesPad(13, ports[2])) == 2);

  // Two leave and come back in swapped order: each finds its own player.
  t.Disconnect(2);
  t.Disconnect(4);
  CHECK(t.Connect(SeriesPad(14, ports[3])) == 3);
  CHECK(t.Connect(SeriesPad(12, ports[1])) == 1);

  // All four out, back in reverse order.
  for (uint64_t id : {1, 12, 13, 14}) {
    t.Disconnect(id);
  }
  CHECK(t.Connect(SeriesPad(24, ports[3])) == 3);
  CHECK(t.Connect(SeriesPad(23, ports[2])) == 2);
  CHECK(t.Connect(SeriesPad(22, ports[1])) == 1);
  CHECK(t.Connect(SeriesPad(21, ports[0])) == 0);
  CHECK(t.table.Reservations().empty());
}

TEST_CASE("A pad replugged into another port falls back to connection order",
          "[input][slots][ids]") {
  Table t;
  t.Connect(SeriesPad(1, "14100000"));
  t.Connect(SeriesPad(2, "14200000"));
  t.Connect(SeriesPad(3, "14300000"));
  t.Disconnect(2);
  // Same model, new port: it still gets the player an identical pad left.
  CHECK(t.Connect(SeriesPad(12, "14400000")) == 1);
  CHECK(t.table.Reservations().empty());

  // A pad whose port could not be read this time also finds it.
  t.Disconnect(3);
  CHECK(t.Connect(SeriesPad(13)) == 2);
}

TEST_CASE("Serial pads ignore port fallback", "[input][slots][ids]") {
  Table t;
  t.Connect(Pad(1, "A"));
  t.Connect(SeriesPad(2, "14200000"));
  t.Disconnect(1);
  // A serial-less pad of another model does not take a serial pad's player.
  CHECK(t.Connect(SeriesPad(3, "14300000")) == 2);
  // A serial pad does not take a serial-less pad's player by model either.
  t.Disconnect(2);
  DeviceInfo same_model_with_serial = SeriesPad(4, "14200000");
  same_model_with_serial.serial = "S";
  CHECK(t.Connect(same_model_with_serial) == 3);
  CHECK(t.Connect(Pad(5, "A")) == 0);
}

TEST_CASE("Pins match a port", "[input][slots][ids]") {
  const DeviceInfo pad = SeriesPad(1, "14200000");
  CHECK(rex::input::PinMatches("14200000", pad));
  CHECK(rex::input::PinMatches("port:14200000", pad));
  CHECK(rex::input::PinMatches("PORT:045E:0B12@14200000", pad));
  CHECK_FALSE(rex::input::PinMatches("port:14300000", pad));
  CHECK_FALSE(rex::input::PinMatches("14200000", SeriesPad(2)));

  Table t;
  t.table.SetPins(rex::input::ParseSlotPins("1=port:14400000;4=port:14100000"));
  CHECK(t.Connect(SeriesPad(1, "14100000")) == 3);
  CHECK(t.Connect(SeriesPad(2, "14200000")) == 1);
  CHECK(t.Connect(SeriesPad(3, "14400000")) == 0);
}

TEST_CASE("Placement order sorts only port-known pads", "[input][slots][ids]") {
  std::vector<DeviceInfo> pads = {SeriesPad(1, "14300000"), Pad(2, "A"), SeriesPad(3, "14100000"),
                                  SeriesPad(4), SeriesPad(5, "14200000")};
  CHECK(rex::input::PlacementOrder(pads) == std::vector<size_t>{2, 1, 4, 3, 0});
}

TEST_CASE("Relaunch: identical pads get the same players from their ports",
          "[input][slots][ids]") {
  const char* ports[] = {"14100000", "14200000", "14300000", "14400000"};
  auto launch = [&](std::vector<int> enumeration) {
    InputSystem input(nullptr);
    auto driver_owner = std::make_unique<IdentityDriver>();
    IdentityDriver* driver = driver_owner.get();
    input.AddDriver(std::move(driver_owner));
    input.SetDeviceAssignment(std::make_unique<SlotAssignment>());
    // Every pad is already plugged in at launch, in the OS's order.
    uint64_t id = 100;
    for (int port : enumeration) {
      driver->Plug(SeriesPad(id++, ports[port]));
    }
    std::map<std::string, int32_t> by_port;
    for (const auto& snap : input.SnapshotDevices()) {
      by_port[snap.info.location] = snap.user;
    }
    return by_port;
  };
  const std::map<std::string, int32_t> expected = {
      {"14100000", 0}, {"14200000", 1}, {"14300000", 2}, {"14400000", 3}};
  CHECK(launch({0, 1, 2, 3}) == expected);
  CHECK(launch({3, 1, 0, 2}) == expected);
  CHECK(launch({2, 3, 1, 0}) == expected);
}

TEST_CASE("Hot-plug of one of four identical pads through InputSystem", "[input][slots][ids]") {
  InputSystem input(nullptr);
  auto driver_owner = std::make_unique<IdentityDriver>();
  IdentityDriver* driver = driver_owner.get();
  input.AddDriver(std::move(driver_owner));
  input.SetDeviceAssignment(std::make_unique<SlotAssignment>());
  driver->Plug(SeriesPad(1, "14300000"));
  driver->Plug(SeriesPad(2, "14100000"));
  driver->Plug(SeriesPad(3, "14400000"));
  driver->Plug(SeriesPad(4, "14200000"));
  CHECK(Players(input) == std::map<uint64_t, int32_t>{{1, 2}, {2, 0}, {3, 3}, {4, 1}});

  driver->Unplug(4);
  CHECK(Players(input) == std::map<uint64_t, int32_t>{{1, 2}, {2, 0}, {3, 3}});
  driver->Unplug(1);
  CHECK(Players(input) == std::map<uint64_t, int32_t>{{2, 0}, {3, 3}});
  // Back in swapped order, one poll apart.
  driver->Plug(SeriesPad(11, "14300000"));
  CHECK(Players(input)[11] == 2);
  driver->Plug(SeriesPad(14, "14200000"));
  CHECK(Players(input) == std::map<uint64_t, int32_t>{{2, 0}, {3, 3}, {11, 2}, {14, 1}});
  CHECK(driver->lights[static_cast<DeviceId>(14)] == 1);
}

// Four identical Xbox Series pads over Bluetooth on macOS: no serial, no USB
// port, one shared GUID (the GUID below is from a real pad's rex-padtest run).

namespace {

DeviceInfo BtPad(uint64_t id, std::string address = "") {
  DeviceInfo info;
  info.id = static_cast<DeviceId>(id);
  info.name = "Xbox Series X Controller";
  info.guid = "0500509d5e040000130b00003ba36d04";
  info.vendor_id = 0x045e;
  info.product_id = 0x0b13;
  info.bt_address = std::move(address);
  return info;
}

const char* kBtAddresses[] = {"98:7a:14:00:00:01", "98:7a:14:00:00:02", "98:7a:14:00:00:03",
                              "98:7a:14:00:00:04"};

}  // namespace

TEST_CASE("Bluetooth address normalises and sits after port in the chain",
          "[input][slots][ids][bt]") {
  using rex::input::NormalizeBluetoothAddress;
  using rex::input::IdentitySource;
  using rex::input::ResolveDeviceIdentity;
  CHECK(NormalizeBluetoothAddress("F0-B1-7A-21-4C-39") == "f0:b1:7a:21:4c:39");
  CHECK(NormalizeBluetoothAddress("f0:b1:7a:21:4c:39") == "f0:b1:7a:21:4c:39");
  CHECK(NormalizeBluetoothAddress("F0B17A214C39") == "f0:b1:7a:21:4c:39");
  CHECK(NormalizeBluetoothAddress("A886814566F88664").empty());
  CHECK(NormalizeBluetoothAddress("00-00-00-00-00-00").empty());
  CHECK(NormalizeBluetoothAddress("").empty());

  const auto bt = ResolveDeviceIdentity(BtPad(1, kBtAddresses[0]));
  CHECK(bt.source == IdentitySource::kBluetooth);
  CHECK(bt.key == "bt:98:7a:14:00:00:01");
  CHECK(bt.unique);
  CHECK(std::string(rex::input::IdentitySourceName(bt.source)) == "bluetooth");

  DeviceInfo with_serial = BtPad(2, kBtAddresses[0]);
  with_serial.serial = "3039363131";
  CHECK(ResolveDeviceIdentity(with_serial).source == IdentitySource::kSerial);
  DeviceInfo with_port = BtPad(3, kBtAddresses[0]);
  with_port.location = "14200000";
  CHECK(ResolveDeviceIdentity(with_port).source == IdentitySource::kLocation);

  DeviceInfo unknown = BtPad(4);
  unknown.location_note = "GCController has no HID services";
  const auto fallback = ResolveDeviceIdentity(unknown);
  CHECK(fallback.source == IdentitySource::kGuid);
  CHECK(fallback.reason.find("no port or Bluetooth address") != std::string::npos);
}

TEST_CASE("Four identical Bluetooth pads keep their players", "[input][slots][ids][bt]") {
  Table t;
  for (uint64_t i = 1; i <= 4; i++) {
    CHECK(t.Connect(BtPad(i, kBtAddresses[i - 1])) == i - 1);
  }
  CHECK(rex::input::DeviceIdentityKey(t.live[0]) != rex::input::DeviceIdentityKey(t.live[1]));

  // One drops (battery, out of range) and reconnects.
  t.Disconnect(3);
  CHECK(t.Connect(BtPad(13, kBtAddresses[2])) == 2);

  // Two drop and reconnect in the other order.
  t.Disconnect(2);
  t.Disconnect(4);
  CHECK(t.Connect(BtPad(14, kBtAddresses[3])) == 3);
  CHECK(t.Connect(BtPad(12, kBtAddresses[1])) == 1);

  // All four off, back on in reverse order.
  for (uint64_t id : {1, 12, 13, 14}) {
    t.Disconnect(id);
  }
  CHECK(t.Connect(BtPad(24, kBtAddresses[3])) == 3);
  CHECK(t.Connect(BtPad(23, kBtAddresses[2])) == 2);
  CHECK(t.Connect(BtPad(22, kBtAddresses[1])) == 1);
  CHECK(t.Connect(BtPad(21, kBtAddresses[0])) == 0);
  CHECK(t.table.Reservations().empty());
}

TEST_CASE("Without addresses, four Bluetooth pads swap like before", "[input][slots][ids][bt]") {
  Table t;
  for (uint64_t i = 1; i <= 4; i++) {
    t.Connect(BtPad(i));
  }
  t.Disconnect(2);
  t.Disconnect(4);
  // Only connection order: the first back takes the oldest waiting player.
  CHECK(t.Connect(BtPad(14)) == 1);
  CHECK(t.Connect(BtPad(12)) == 3);
}

TEST_CASE("Pins match a Bluetooth address", "[input][slots][ids][bt]") {
  const DeviceInfo pad = BtPad(1, kBtAddresses[2]);
  CHECK(rex::input::PinMatches("bt:98:7a:14:00:00:03", pad));
  CHECK(rex::input::PinMatches("98:7A:14:00:00:03", pad));
  CHECK_FALSE(rex::input::PinMatches("bt:98:7a:14:00:00:04", pad));
}

TEST_CASE("Relaunch: identical Bluetooth pads get the same players from their addresses",
          "[input][slots][ids][bt]") {
  auto launch = [&](std::vector<int> enumeration) {
    InputSystem input(nullptr);
    auto driver_owner = std::make_unique<IdentityDriver>();
    IdentityDriver* driver = driver_owner.get();
    input.AddDriver(std::move(driver_owner));
    input.SetDeviceAssignment(std::make_unique<SlotAssignment>());
    uint64_t id = 100;
    for (int k : enumeration) {
      driver->Plug(BtPad(id++, kBtAddresses[k]));
    }
    std::map<std::string, int32_t> by_address;
    for (const auto& snap : input.SnapshotDevices()) {
      by_address[snap.info.bt_address] = snap.user;
    }
    return by_address;
  };
  const std::map<std::string, int32_t> expected = {{kBtAddresses[0], 0},
                                                   {kBtAddresses[1], 1},
                                                   {kBtAddresses[2], 2},
                                                   {kBtAddresses[3], 3}};
  CHECK(launch({0, 1, 2, 3}) == expected);
  CHECK(launch({3, 1, 0, 2}) == expected);
  CHECK(launch({2, 3, 1, 0}) == expected);
}

TEST_CASE("Placement order: USB pads by port, then Bluetooth pads by address",
          "[input][slots][ids][bt]") {
  std::vector<DeviceInfo> pads = {BtPad(1, kBtAddresses[1]), SeriesPad(2, "14200000"),
                                  BtPad(3, kBtAddresses[0]), SeriesPad(4, "14100000")};
  CHECK(rex::input::PlacementOrder(pads) == std::vector<size_t>{3, 1, 2, 0});
}

TEST_CASE("Hot-plug of one of four Bluetooth pads through InputSystem",
          "[input][slots][ids][bt]") {
  InputSystem input(nullptr);
  auto driver_owner = std::make_unique<IdentityDriver>();
  IdentityDriver* driver = driver_owner.get();
  input.AddDriver(std::move(driver_owner));
  input.SetDeviceAssignment(std::make_unique<SlotAssignment>());
  for (uint64_t i = 1; i <= 4; i++) {
    driver->Plug(BtPad(i, kBtAddresses[i - 1]));
  }
  CHECK(Players(input) == std::map<uint64_t, int32_t>{{1, 0}, {2, 1}, {3, 2}, {4, 3}});
  driver->Unplug(2);
  driver->Unplug(3);
  driver->Plug(BtPad(13, kBtAddresses[2]));
  driver->Plug(BtPad(12, kBtAddresses[1]));
  CHECK(Players(input) == std::map<uint64_t, int32_t>{{1, 0}, {4, 3}, {12, 1}, {13, 2}});
}
