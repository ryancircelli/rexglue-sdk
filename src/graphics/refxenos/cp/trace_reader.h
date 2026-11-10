// refxenos: Xenia-format .xtr trace reader.
//
// Parses the whole file at Open() into a flat command list whose payload
// pointers alias the file bytes held by the reader. Decode() inflates
// MemoryRead / Registers / GammaRamp payloads (kNone or ReXGlue's zlib
// extension, encoding 2).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rex/graphics/trace/trace_protocol.h>

namespace refxenos {

namespace trace = rex::graphics::trace;

// MemoryEncodingFormat values. The SDK header only names kNone and kSnappy;
// kZlib = 2 is the ReXGlue extension written by the gpu-test-mftrace capture.
enum class TraceEncoding : uint32_t {
  kNone = 0,
  kSnappy = 1,
  kZlib = 2,
};

struct TraceCommand {
  trace::TraceCommandType type{};
  size_t file_offset = 0;  // byte offset of the command header in the file

  // PrimaryBufferStart / IndirectBufferStart / PacketStart / MemoryRead /
  // MemoryWrite.
  uint32_t base_ptr = 0;
  // Buffer/packet starts: dword count. Registers: register count.
  uint32_t count = 0;

  // Payload that follows the fixed header (packet dwords, encoded memory,
  // encoded registers, encoded gamma ramp, encoded EDRAM snapshot).
  const uint8_t* payload = nullptr;
  uint32_t payload_bytes = 0;
  TraceEncoding encoding = TraceEncoding::kNone;
  uint32_t decoded_length = 0;  // MemoryRead/MemoryWrite only

  // Registers.
  uint32_t first_register = 0;
  bool execute_callbacks = false;
  // Event.
  uint32_t event_type = 0;  // 0 = kSwap
  // GammaRamp.
  uint8_t rw_component = 0;
};

class TraceReader {
 public:
  bool Open(const std::string& path);
  // Parses an in-memory image (used by tests); the reader keeps a copy.
  bool OpenBytes(std::vector<uint8_t> bytes);

  const std::string& error() const { return error_; }
  uint32_t version() const { return version_; }
  uint32_t title_id() const { return title_id_; }
  // build_commit_sha with trailing NULs removed (40 chars at most).
  const std::string& build_sha() const { return build_sha_; }
  const std::vector<TraceCommand>& commands() const { return commands_; }
  size_t file_size() const { return data_.size(); }

  // Decodes the payload of a MemoryRead / MemoryWrite / Registers / GammaRamp
  // / EdramSnapshot command into `out`. For memory commands the result is
  // decoded_length bytes; for the others it is whatever the payload inflates
  // to. Returns false (and sets *err) for snappy or corrupt blocks.
  static bool Decode(const TraceCommand& cmd, std::vector<uint8_t>& out, std::string* err);

 private:
  bool Parse();

  std::vector<uint8_t> data_;
  std::vector<TraceCommand> commands_;
  std::string error_;
  uint32_t version_ = 0;
  uint32_t title_id_ = 0;
  std::string build_sha_;
};

}  // namespace refxenos
