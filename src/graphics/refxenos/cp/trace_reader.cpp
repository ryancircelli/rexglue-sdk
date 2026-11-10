#include "cp/trace_reader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

namespace refxenos {

namespace {

template <typename T>
bool ReadStruct(const uint8_t* data, size_t size, size_t offset, T& out) {
  if (offset > size || size - offset < sizeof(T)) return false;
  std::memcpy(&out, data + offset, sizeof(T));
  return true;
}

// Inflates a zlib stream. If expected != 0 the output must be exactly that
// size; otherwise the buffer grows until the stream ends.
bool Inflate(const uint8_t* src, uint32_t src_len, uint32_t expected, std::vector<uint8_t>& out,
             std::string* err) {
  z_stream zs{};
  if (inflateInit(&zs) != Z_OK) {
    if (err) *err = "inflateInit failed";
    return false;
  }
  out.assign(expected ? expected : std::max<size_t>(size_t(src_len) * 4, 4096), 0);
  zs.next_in = const_cast<Bytef*>(src);
  zs.avail_in = src_len;
  size_t produced = 0;
  int rc = Z_OK;
  while (true) {
    zs.next_out = out.data() + produced;
    zs.avail_out = uInt(out.size() - produced);
    rc = inflate(&zs, Z_NO_FLUSH);
    produced = out.size() - zs.avail_out;
    if (rc == Z_STREAM_END) break;
    if (rc != Z_OK && rc != Z_BUF_ERROR) break;
    if (zs.avail_out == 0) {
      if (expected) {
        // More data than declared.
        rc = Z_DATA_ERROR;
        break;
      }
      out.resize(out.size() * 2);
      continue;
    }
    if (zs.avail_in == 0) {
      rc = Z_DATA_ERROR;  // truncated stream
      break;
    }
  }
  inflateEnd(&zs);
  if (rc != Z_STREAM_END) {
    if (err) *err = "zlib inflate failed (rc " + std::to_string(rc) + ")";
    return false;
  }
  if (expected && produced != expected) {
    if (err) {
      *err = "zlib block decoded to " + std::to_string(produced) + " bytes, expected " +
             std::to_string(expected);
    }
    return false;
  }
  out.resize(produced);
  return true;
}

}  // namespace

TraceReader::~TraceReader() { Close(); }

void TraceReader::Close() {
  if (map_) munmap(map_, size_);
  map_ = nullptr;
  owned_.clear();
  data_ = nullptr;
  size_ = 0;
  commands_.clear();
  error_.clear();
  truncated_ = false;
  truncated_at_ = 0;
}

bool TraceReader::Open(const std::string& path) {
  Close();
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    error_ = "cannot open " + path;
    return false;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    ::close(fd);
    error_ = "cannot stat " + path;
    return false;
  }
  size_ = size_t(st.st_size);
  if (size_) {
    void* m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) {
      ::close(fd);
      error_ = "mmap failed on " + path;
      size_ = 0;
      return false;
    }
    map_ = m;
    data_ = static_cast<const uint8_t*>(m);
  }
  ::close(fd);
  return Parse();
}

bool TraceReader::OpenBytes(std::vector<uint8_t> bytes) {
  Close();
  owned_ = std::move(bytes);
  data_ = owned_.data();
  size_ = owned_.size();
  return Parse();
}

bool TraceReader::Parse() {
  trace::TraceHeader header;
  if (!ReadStruct(data_, size_, 0, header)) {
    error_ = "file too small for TraceHeader";
    return false;
  }
  version_ = header.version;
  title_id_ = header.title_id;
  size_t sha_len = 0;
  while (sha_len < sizeof(header.build_commit_sha) && header.build_commit_sha[sha_len]) ++sha_len;
  build_sha_.assign(header.build_commit_sha, sha_len);
  if (version_ != trace::kTraceFormatVersion) {
    error_ = "unsupported trace version " + std::to_string(version_);
    return false;
  }

  using T = trace::TraceCommandType;
  size_t off = sizeof(trace::TraceHeader);
  // A command cut short by the end of the file ends the trace cleanly.
  auto truncate = [&](size_t at) {
    truncated_ = true;
    truncated_at_ = at;
    error_ = "trace truncated at byte " + std::to_string(at) + " of " + std::to_string(size_);
    return true;
  };
  truncated_at_ = size_;
  while (off < size_) {
    uint32_t raw_type;
    if (!ReadStruct(data_, size_, off, raw_type)) return truncate(off);
    TraceCommand cmd;
    cmd.type = T(raw_type);
    cmd.file_offset = off;
    auto payload = [&](size_t header_size, uint32_t bytes) -> bool {
      size_t start = off + header_size;
      if (start > size_ || size_ - start < bytes) {
        return false;
      }
      cmd.payload = data_ + start;
      cmd.payload_bytes = bytes;
      off = start + bytes;
      return true;
    };
    auto fixed = [&](auto& s) -> bool {
      if (!ReadStruct(data_, size_, off, s)) {
        return false;
      }
      return true;
    };
    switch (cmd.type) {
      case T::kPrimaryBufferStart: {
        trace::PrimaryBufferStartCommand c;
        if (!fixed(c)) return truncate(off);
        cmd.base_ptr = c.base_ptr;
        cmd.count = c.count;
        off += sizeof(c);
      } break;
      case T::kPrimaryBufferEnd:
        if (size_ - off < sizeof(trace::PrimaryBufferEndCommand)) return truncate(off);
        off += sizeof(trace::PrimaryBufferEndCommand);
        break;
      case T::kIndirectBufferStart: {
        trace::IndirectBufferStartCommand c;
        if (!fixed(c)) return truncate(off);
        cmd.base_ptr = c.base_ptr;
        cmd.count = c.count;
        off += sizeof(c);
      } break;
      case T::kIndirectBufferEnd:
        if (size_ - off < sizeof(trace::IndirectBufferEndCommand)) return truncate(off);
        off += sizeof(trace::IndirectBufferEndCommand);
        break;
      case T::kPacketStart: {
        trace::PacketStartCommand c;
        if (!fixed(c)) return truncate(off);
        cmd.base_ptr = c.base_ptr;
        cmd.count = c.count;
        if (!payload(sizeof(c), c.count * 4u)) return truncate(off);
      } break;
      case T::kPacketEnd:
        if (size_ - off < sizeof(trace::PacketEndCommand)) return truncate(off);
        off += sizeof(trace::PacketEndCommand);
        break;
      case T::kMemoryRead:
      case T::kMemoryWrite: {
        trace::MemoryCommand c;
        if (!fixed(c)) return truncate(off);
        cmd.base_ptr = c.base_ptr;
        cmd.encoding = TraceEncoding(uint32_t(c.encoding_format));
        cmd.decoded_length = c.decoded_length;
        if (!payload(sizeof(c), c.encoded_length)) return truncate(off);
      } break;
      case T::kEdramSnapshot: {
        trace::EdramSnapshotCommand c;
        if (!fixed(c)) return truncate(off);
        cmd.encoding = TraceEncoding(uint32_t(c.encoding_format));
        if (!payload(sizeof(c), c.encoded_length)) return truncate(off);
      } break;
      case T::kEvent: {
        trace::EventCommand c;
        if (!fixed(c)) return truncate(off);
        cmd.event_type = uint32_t(c.event_type);
        off += sizeof(c);
      } break;
      case T::kRegisters: {
        trace::RegistersCommand c;
        if (!fixed(c)) return truncate(off);
        cmd.first_register = c.first_register;
        cmd.count = c.register_count;
        cmd.execute_callbacks = c.execute_callbacks;
        cmd.encoding = TraceEncoding(uint32_t(c.encoding_format));
        if (!payload(sizeof(c), c.encoded_length)) return truncate(off);
      } break;
      case T::kGammaRamp: {
        trace::GammaRampCommand c;
        if (!fixed(c)) return truncate(off);
        cmd.rw_component = c.rw_component;
        cmd.encoding = TraceEncoding(uint32_t(c.encoding_format));
        if (!payload(sizeof(c), c.encoded_length)) return truncate(off);
      } break;
      default:
        error_ = "unknown trace command " + std::to_string(raw_type) + " at offset " +
                 std::to_string(off);
        return false;
    }
    commands_.push_back(cmd);
  }
  return true;
}

bool TraceReader::Decode(const TraceCommand& cmd, std::vector<uint8_t>& out, std::string* err) {
  using T = trace::TraceCommandType;
  uint32_t expected = 0;
  if (cmd.type == T::kMemoryRead || cmd.type == T::kMemoryWrite) {
    expected = cmd.decoded_length;
  } else if (cmd.type == T::kRegisters) {
    expected = cmd.count * 4u;
  }
  switch (cmd.encoding) {
    case TraceEncoding::kNone: {
      uint32_t n = cmd.payload_bytes;
      if (expected && expected < n) n = expected;
      out.assign(cmd.payload, cmd.payload + n);
      if (expected && n < expected) {
        if (err) *err = "uncompressed block shorter than its decoded length";
        return false;
      }
      return true;
    }
    case TraceEncoding::kZlib:
      return Inflate(cmd.payload, cmd.payload_bytes, expected, out, err);
    case TraceEncoding::kSnappy:
      if (err) *err = "snappy-encoded blocks are not supported";
      return false;
  }
  if (err) *err = "unknown block encoding " + std::to_string(uint32_t(cmd.encoding));
  return false;
}

}  // namespace refxenos
