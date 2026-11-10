// Golden store implementation. Spec: bo2-mac-port/research/golden-store.md v1.1
#include "golden_store.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <set>

namespace refxenos::golden {

namespace fs = std::filesystem;
using nlohmann::json;

// ---------------------------------------------------------------- SHA-256
namespace {

constexpr uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void Compress(uint32_t h[8], const uint8_t* p) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i)
    w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
           (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
  for (int i = 16; i < 64; ++i) {
    uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6],
           hh = h[7];
  for (int i = 0; i < 64; ++i) {
    uint32_t S1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = hh + S1 + ch + kK[i] + w[i];
    uint32_t S0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
    uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + mj;
    hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d;
  h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

}  // namespace

std::string Sha256Hex(const void* data, size_t size) {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const uint8_t* p = static_cast<const uint8_t*>(data);
  size_t full = size / 64;
  for (size_t i = 0; i < full; ++i) Compress(h, p + 64 * i);
  uint8_t tail[128] = {};
  size_t rem = size - full * 64;
  if (rem) std::memcpy(tail, p + full * 64, rem);
  tail[rem] = 0x80;
  size_t tail_len = (rem < 56) ? 64 : 128;
  uint64_t bits = uint64_t(size) * 8;
  for (int i = 0; i < 8; ++i) tail[tail_len - 1 - i] = uint8_t(bits >> (8 * i));
  for (size_t off = 0; off < tail_len; off += 64) Compress(h, tail + off);
  static const char* hex = "0123456789abcdef";
  std::string out(64, '0');
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) {
      uint8_t b = uint8_t(h[i] >> (24 - 8 * j));
      out[i * 8 + j * 2] = hex[b >> 4];
      out[i * 8 + j * 2 + 1] = hex[b & 15];
    }
  return out;
}

// ---------------------------------------------------------- canonical JSON
namespace {

void AppendU16(std::string* out, unsigned v) {
  static const char* hex = "0123456789abcdef";
  *out += "\\u";
  for (int s = 12; s >= 0; s -= 4) *out += hex[(v >> s) & 15];
}

// Python json ensure_ascii=True string encoding.
void AppendString(std::string* out, const std::string& s) {
  *out += '"';
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    if (c < 0x80) {
      switch (c) {
        case '"': *out += "\\\""; break;
        case '\\': *out += "\\\\"; break;
        case '\n': *out += "\\n"; break;
        case '\r': *out += "\\r"; break;
        case '\t': *out += "\\t"; break;
        case '\b': *out += "\\b"; break;
        case '\f': *out += "\\f"; break;
        default:
          if (c < 0x20 || c == 0x7f) AppendU16(out, c);
          else *out += char(c);  // Python escapes anything outside " "..."~" (so 0x7f too)
      }
      ++i;
      continue;
    }
    // Decode UTF-8 (nlohmann guarantees valid UTF-8 for parsed input).
    uint32_t cp;
    int n;
    if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 2; }
    else { cp = c & 0x07; n = 3; }
    for (int k = 1; k <= n && i + k < s.size(); ++k)
      cp = (cp << 6) | (uint8_t(s[i + k]) & 0x3F);
    i += n + 1;
    if (cp >= 0x10000) {  // surrogate pair, as Python does
      cp -= 0x10000;
      AppendU16(out, 0xD800 + (cp >> 10));
      AppendU16(out, 0xDC00 + (cp & 0x3FF));
    } else {
      AppendU16(out, cp);
    }
  }
  *out += '"';
}

// Python float.__repr__: shortest round-trip digits; scientific when the
// decimal exponent is < -4 or >= 16, otherwise positional with ".0" if whole.
void AppendDouble(std::string* out, double d) {
  if (std::isnan(d)) { *out += "NaN"; return; }  // Python json default
  if (std::isinf(d)) { *out += d < 0 ? "-Infinity" : "Infinity"; return; }
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
  std::string s(buf, r.ptr);  // e.g. "-1.2345e+02", "5e-324"
  bool neg = false;
  size_t pos = 0;
  if (s[0] == '-') { neg = true; pos = 1; }
  size_t epos = s.find('e');
  std::string mant = s.substr(pos, epos - pos);
  int exp10 = std::atoi(s.c_str() + epos + 1);
  std::string digits;
  for (char c : mant) if (c != '.') digits += c;
  if (neg) *out += '-';
  if (exp10 < -4 || exp10 >= 16) {
    *out += digits[0];
    if (digits.size() > 1) { *out += '.'; out->append(digits, 1, std::string::npos); }
    char eb[16];
    std::snprintf(eb, sizeof eb, "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
    *out += eb;
  } else if (exp10 < 0) {
    *out += "0.";
    out->append(size_t(-exp10 - 1), '0');
    *out += digits;
  } else {
    size_t int_len = size_t(exp10) + 1;
    if (digits.size() <= int_len) {
      *out += digits;
      out->append(int_len - digits.size(), '0');
      *out += ".0";
    } else {
      out->append(digits, 0, int_len);
      *out += '.';
      out->append(digits, int_len, std::string::npos);
    }
  }
}

void Canon(std::string* out, const json& j) {
  switch (j.type()) {
    case json::value_t::null: *out += "null"; break;
    case json::value_t::boolean: *out += j.get<bool>() ? "true" : "false"; break;
    case json::value_t::number_integer: *out += std::to_string(j.get<int64_t>()); break;
    case json::value_t::number_unsigned: *out += std::to_string(j.get<uint64_t>()); break;
    case json::value_t::number_float: AppendDouble(out, j.get<double>()); break;
    case json::value_t::string: AppendString(out, j.get_ref<const std::string&>()); break;
    case json::value_t::array: {
      *out += '[';
      bool first = true;
      for (const auto& e : j) {
        if (!first) *out += ',';
        first = false;
        Canon(out, e);
      }
      *out += ']';
      break;
    }
    case json::value_t::object: {
      // nlohmann's default object is std::map: keys iterate in UTF-8 byte
      // order == Python's code point order, so sort_keys=True is implicit.
      *out += '{';
      bool first = true;
      for (auto it = j.begin(); it != j.end(); ++it) {
        if (!first) *out += ',';
        first = false;
        AppendString(out, it.key());
        *out += ':';
        Canon(out, it.value());
      }
      *out += '}';
      break;
    }
    default: *out += "null"; break;  // binary/discarded: not valid in manifests
  }
}

// ------------------------------------------------------------- file helpers

std::string TempSuffix() {
  static std::atomic<uint64_t> counter{0};
  static const uint64_t salt = std::random_device{}();
  return ".tmp." + std::to_string(::getpid()) + "." + std::to_string(salt) +
         "." + std::to_string(counter++);
}

bool WriteFileAtomic(const fs::path& path, const void* data, size_t size) {
  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  fs::path tmp = path;
  tmp += TempSuffix();
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(static_cast<const char*>(data), std::streamsize(size));
    f.flush();
    if (!f) { fs::remove(tmp, ec); return false; }
  }
  fs::rename(tmp, path, ec);  // atomic within one filesystem
  if (ec) { fs::remove(tmp, ec); return false; }
  return true;
}

bool ReadFile(const fs::path& path, std::vector<uint8_t>* out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  auto n = f.tellg();
  out->resize(size_t(n));
  f.seekg(0);
  if (n > 0) f.read(reinterpret_cast<char*>(out->data()), n);
  return bool(f);
}

bool ReadJsonFile(const fs::path& path, json* out) {
  std::vector<uint8_t> b;
  if (!ReadFile(path, &b)) return false;
  *out = json::parse(b.begin(), b.end(), nullptr, /*allow_exceptions=*/false);
  return !out->is_discarded();
}

bool WriteJsonFile(const fs::path& path, const json& j) {
  std::string s = j.dump(2) + "\n";
  return WriteFileAtomic(path, s.data(), s.size());
}

// RAII flock on a dedicated lock file (the manifest itself is replaced by
// rename, so locking it directly would not serialize writers).
class FileLock {
 public:
  explicit FileLock(const fs::path& lock_path) {
    std::error_code ec;
    fs::create_directories(lock_path.parent_path(), ec);
    fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ >= 0) ::flock(fd_, LOCK_EX);
  }
  ~FileLock() {
    if (fd_ >= 0) { ::flock(fd_, LOCK_UN); ::close(fd_); }
  }
  bool ok() const { return fd_ >= 0; }
 private:
  int fd_ = -1;
};

bool IsHex64(const std::string& h) {
  if (h.size() != 64) return false;
  for (char c : h) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}

bool SafeName(const std::string& s) {  // layer / id used as path components
  if (s.empty() || s == "." || s == "..") return false;
  return s.find('/') == std::string::npos && s.find('\\') == std::string::npos;
}

}  // namespace

std::string CanonicalJson(const json& j) {
  std::string s;
  Canon(&s, j);
  return s;
}

// ------------------------------------------------------------------ Store
Store::Store(fs::path root) : root_(std::move(root)) {}

fs::path Store::ObjectPath(const std::string& h) const {
  return root_ / "objects" / h.substr(0, 2) / h;
}
fs::path Store::CasePath(const std::string& l, const std::string& id) const {
  return root_ / "cases" / l / (id + ".json");
}
fs::path Store::DisputePath(const std::string& l, const std::string& id) const {
  return root_ / "disputes" / l / (id + ".json");
}

// Spec "Layout": blobs raw, temp file in the same directory then rename, never
// rewrite an existing object.
std::string Store::PutObject(const void* data, size_t size) {
  static const uint8_t kEmpty = 0;
  if (!data) data = &kEmpty;
  std::string h = Sha256Hex(data, size);
  fs::path p = ObjectPath(h);
  std::error_code ec;
  if (fs::exists(p, ec)) return h;
  if (!WriteFileAtomic(p, data, size)) return std::string();
  return h;
}

bool Store::HasObject(const std::string& h) const {
  if (!IsHex64(h)) return false;
  std::error_code ec;
  return fs::exists(ObjectPath(h), ec);
}

bool Store::GetObject(const std::string& h, std::vector<uint8_t>* out) const {
  if (!IsHex64(h)) return false;
  return ReadFile(ObjectPath(h), out);
}

// Spec "Layout": case_id = sha256 of canonical JSON of {"layer","inputs","params"}.
std::string Store::CaseId(const json& c) const {
  json k = json::object();
  k["layer"] = c.value("layer", json());
  k["inputs"] = c.contains("inputs") ? c["inputs"] : json::object();
  k["params"] = c.contains("params") ? c["params"] : json::object();
  std::string s = CanonicalJson(k);
  return Sha256Hex(s.data(), s.size());
}

bool Store::ReadCase(const std::string& layer, const std::string& id,
                     json* out) const {
  if (!SafeName(layer) || !SafeName(id)) return false;
  return ReadJsonFile(CasePath(layer, id), out) ||
         ReadJsonFile(DisputePath(layer, id), out);
}

// ----------------------------------------------------------------- rules
bool Store::LoadRules(json* out) const {
  fs::path p = root_ / "rules.json";
  std::error_code ec;
  if (!fs::exists(p, ec)) { *out = json::object(); return true; }
  return ReadJsonFile(p, out) && out->is_object();
}

std::vector<std::string> Store::StaleRules(const json& e, const json& reg) const {
  std::vector<std::string> stale;
  if (!e.is_object() || !e.contains("rules") || !e["rules"].is_object()) return stale;
  for (auto it = e["rules"].begin(); it != e["rules"].end(); ++it) {
    auto r = reg.find(it.key());
    if (r == reg.end() || !r->is_object() || !r->contains("version")) continue;
    if (it.value().is_number() && it.value().get<int64_t>() < (*r)["version"].get<int64_t>())
      stale.push_back(it.key());
  }
  return stale;
}

bool Store::IsEntryStale(const json& e, const json& reg) const {
  return !StaleRules(e, reg).empty();
}

bool Store::IsEntryStale(const json& e) const {
  json reg;
  if (!LoadRules(&reg)) return false;
  return IsEntryStale(e, reg);
}

bool Store::RegisterRule(const std::string& id, int64_t version,
                         const std::string& source, bool assumed) {
  FileLock lock(root_ / "rules.json.lock");
  if (!lock.ok()) return false;
  json reg;
  if (!LoadRules(&reg)) return false;
  auto it = reg.find(id);
  if (it != reg.end() && it->is_object() && it->value("version", int64_t(0)) >= version)
    return it->value("version", int64_t(0)) == version ? true : false;  // never lower
  reg[id] = {{"version", version}, {"source", source}, {"assumed", assumed}};
  return WriteJsonFile(root_ / "rules.json", reg);
}

int64_t Store::BumpRule(const std::string& id) {
  FileLock lock(root_ / "rules.json.lock");
  if (!lock.ok()) return -1;
  json reg;
  if (!LoadRules(&reg) || !reg.contains(id)) return -1;
  int64_t v = reg[id].value("version", int64_t(0)) + 1;
  reg[id]["version"] = v;
  return WriteJsonFile(root_ / "rules.json", reg) ? v : -1;
}

// ---------------------------------------------------------- status/golden
// Spec "Case manifest": agreed = every (non-stale) referee has identical
// hashes for every output; single = one referee; disputed = no golden.
// Spec "Rule versions": stale entries never count toward agreed.
// Decisions where the spec is silent (see report):
//  - outputs whose spec has compare "none" are never compared nor in golden;
//  - only output names present in ALL fresh referees are compared; names
//    present in a subset do not cause a dispute and are left out of golden
//    when 2+ referees are fresh (optional diagnostics like mip<N>_blocks);
//  - zero fresh referees: status "single" with no golden (nothing to compare
//    against); stale-ness is reported by IsEntryStale, not by status.
void RecomputeStatus(json* m, const json& registry) {
  Store dummy{fs::path()};
  std::vector<std::string> fresh;
  for (auto it = (*m)["referees"].begin(); it != (*m)["referees"].end(); ++it)
    if (!dummy.IsEntryStale(it.value(), registry)) fresh.push_back(it.key());

  auto compared = [&](const std::string& name) {
    const json& o = (*m)["outputs"];
    if (o.contains(name) && o[name].is_object() && o[name].value("compare", "bits") == "none")
      return false;
    return true;
  };

  m->erase("golden");
  if (fresh.empty()) { (*m)["status"] = "single"; return; }
  if (fresh.size() == 1) {
    json g = json::object();
    for (auto& [name, h] : (*m)["referees"][fresh[0]]["outputs"].items())
      if (compared(name)) g[name] = h;
    (*m)["status"] = "single";
    (*m)["golden"] = g;
    return;
  }
  // 2+ fresh referees.
  std::set<std::string> common;
  bool first = true;
  for (const auto& r : fresh) {
    std::set<std::string> names;
    for (auto& [name, h] : (*m)["referees"][r]["outputs"].items())
      if (compared(name)) names.insert(name);
    if (first) { common = names; first = false; }
    else {
      std::set<std::string> inter;
      for (auto& n : common) if (names.count(n)) inter.insert(n);
      common = inter;
    }
  }
  bool agree = true;
  json g = json::object();
  for (const auto& name : common) {
    const json& h0 = (*m)["referees"][fresh[0]]["outputs"][name];
    for (size_t i = 1; i < fresh.size(); ++i)
      if ((*m)["referees"][fresh[i]]["outputs"][name] != h0) agree = false;
    g[name] = h0;
  }
  if (agree) {
    (*m)["status"] = "agreed";
    (*m)["golden"] = g;
  } else {
    (*m)["status"] = "disputed";  // no golden
  }
}

// ----------------------------------------------------------- case writing
bool Store::WriteCaseFromReferee(const json& core, const std::string& referee,
                                 const json& entry, std::string* id_out,
                                 std::string* error) {
  auto fail = [&](const std::string& msg) {
    if (error) *error = msg;
    return false;
  };
  if (!core.is_object() || !core.contains("layer") || !core["layer"].is_string() ||
      !core.contains("inputs"))
    return fail("case_core needs layer and inputs");
  if (!entry.is_object() || !entry.contains("outputs") || !entry["outputs"].is_object())
    return fail("referee_entry needs outputs{name:hash}");
  if (referee.empty()) return fail("empty referee name");
  const std::string layer = core["layer"].get<std::string>();
  const std::string id = CaseId(core);
  if (!SafeName(layer)) return fail("bad layer name");
  if (id_out) *id_out = id;

  // Serialize merges of the same case across processes. The lock file lives
  // next to the manifest; different cases never contend.
  FileLock lock(root_ / "cases" / layer / (id + ".lock"));
  if (!lock.ok()) return fail("cannot lock case");

  json m;
  bool existed = ReadCase(layer, id, &m);
  if (!existed) {
    m = json::object();
    m["schema"] = 1;
    m["layer"] = layer;
    m["id"] = id;
    m["game"] = core.value("game", "bo2");
    m["source"] = core.value("source", "");
    m["inputs"] = core["inputs"];
    m["params"] = core.contains("params") ? core["params"] : json::object();
    m["outputs"] = core.contains("outputs") ? core["outputs"] : json::object();
    m["referees"] = json::object();
  } else {
    // Never alter existing fields; only add output specs that are new.
    if (core.contains("outputs") && core["outputs"].is_object())
      for (auto& [name, spec] : core["outputs"].items())
        if (!m["outputs"].contains(name)) m["outputs"][name] = spec;
    if (!m.contains("referees")) m["referees"] = json::object();
  }
  m["referees"][referee] = entry;  // only this referee's entry is touched

  json reg;
  if (!LoadRules(&reg)) return fail("rules.json unreadable");
  RecomputeStatus(&m, reg);

  // Spec: disputed cases live in disputes/ (no golden); otherwise cases/.
  const bool disputed = m["status"] == "disputed";
  fs::path dst = disputed ? DisputePath(layer, id) : CasePath(layer, id);
  fs::path other = disputed ? CasePath(layer, id) : DisputePath(layer, id);
  if (!WriteJsonFile(dst, m)) return fail("cannot write manifest");
  std::error_code ec;
  fs::remove(other, ec);  // move semantics; ignore if absent
  return true;
}

}  // namespace refxenos::golden
