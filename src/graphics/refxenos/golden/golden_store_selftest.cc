// Self test for golden_store. Build:
//   clang++ -std=c++20 -O2 -I<nlohmann include dir> golden_store.cc golden_store_selftest.cc -o selftest
// Uses a temp store under /tmp/bo2/texconf/golden-agent/store (never the real one).
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "golden_store.h"

using namespace refxenos::golden;
using nlohmann::json;
namespace fs = std::filesystem;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define CHECK_EQ(a, b) do { auto va = (a); auto vb = (b); if (!(va == vb)) { \
  std::printf("FAIL %s:%d: %s != %s\n  got:  %s\n  want: %s\n", __FILE__, __LINE__, #a, #b, std::string(va).c_str(), std::string(vb).c_str()); ++g_fail; } } while (0)

static const fs::path kScratch = "/tmp/bo2/texconf/golden-agent";

static std::string Sha(const std::string& s) { return Sha256Hex(s.data(), s.size()); }

static void TestSha() {
  CHECK_EQ(Sha("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK_EQ(Sha(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK_EQ(Sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
           "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  CHECK_EQ(Sha(std::string(1000000, 'a')),
           "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  // Padding boundaries vs python hashlib are covered via objects below; also
  // check lengths 55/56/63/64/65 against known values via python.
  for (size_t n : {55, 56, 63, 64, 65, 119, 120}) {
    std::string s(n, 'x');
    std::string cmd = "python3 -I -c \"import hashlib;print(hashlib.sha256(b'x'*" + std::to_string(n) + ").hexdigest())\"";
    FILE* p = popen(cmd.c_str(), "r");
    char buf[128] = {};
    if (p) { std::fgets(buf, sizeof buf, p); pclose(p); }
    std::string want(buf);
    while (!want.empty() && want.back() == '\n') want.pop_back();
    CHECK_EQ(Sha(s), want);
  }
}

// Python canonical sha256 + canonical string of a JSON text.
static std::string PyCanon(const std::string& text) {
  fs::path f = kScratch / "canon_in.json";
  { std::ofstream o(f, std::ios::binary); o << text; }
  std::string cmd = "python3 -I -c \"import json,sys;print(json.dumps(json.load(open(sys.argv[1],encoding='utf-8')),"
                    "sort_keys=True,separators=(',',':')),end='')\" " + f.string();
  FILE* p = popen(cmd.c_str(), "r");
  std::string out;
  char buf[4096];
  size_t n;
  while (p && (n = std::fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
  if (p) pclose(p);
  return out;
}

static void TestCanonical(Store& st) {
  const char* cases[] = {
      R"({"layer":"texel_decode","inputs":{"b":"x","a":"y"},"params":{"mip":[0,1,2],"slice":[0]}})",
      R"({"z":1,"a":{"y":[1.5,2.0,-0.0,1e-5,1e16,1e15,123456789.125,0.1,1e22,5e-324,1.7976931348623157e308],"x":null},"t":true,"f":false})",
      R"({"s":"café 中文 😀 \"q\" \\ \n\t\r\b\f \u0001 \u007f /","kéy":1})",
      R"({"big":18446744073709551615,"neg":-9223372036854775808,"e":1E2,"f":12.50,"tiny":0.0001,"small":0.00001234,"mid":1234567890123456.0})",
      R"({"params":{"lod_bias":0.5,"offset_x":-1,"unnormalized":false,"sys":{"c1":3.14159,"a":[1,[2,[3.0]]]}}})",
      R"([])", R"({})", R"("str")", R"(7)", R"(0.30000000000000004)"};
  for (const char* c : cases) {
    json j = json::parse(c);
    CHECK_EQ(CanonicalJson(j), PyCanon(c));
  }
  // CaseId = sha256 of canonical {"layer","inputs","params"}; extra keys ignored.
  json core = json::parse(cases[0]);
  core["source"] = "ignored";
  core["outputs"] = json::object();
  std::string want = Sha(PyCanon(cases[0]));
  CHECK_EQ(st.CaseId(core), want);
}

static void TestObjects(Store& st) {
  std::vector<uint8_t> blob(5000);
  for (size_t i = 0; i < blob.size(); ++i) blob[i] = uint8_t(i * 31);
  std::string h = st.PutObject(blob.data(), blob.size());
  CHECK_EQ(h, Sha256Hex(blob.data(), blob.size()));
  CHECK(fs::exists(st.root() / "objects" / h.substr(0, 2) / h));
  std::vector<uint8_t> back;
  CHECK(st.GetObject(h, &back));
  CHECK(back == blob);
  CHECK(st.PutObject(blob.data(), blob.size()) == h);  // skip if exists
  std::string he = st.PutObject(nullptr, 0);
  CHECK_EQ(he, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(st.GetObject(he, &back) && back.empty());
  CHECK(!st.GetObject(std::string(64, 'a'), &back));
  CHECK(!st.GetObject("../../etc/passwd", &back));
  // No temp files left behind.
  for (auto& e : fs::recursive_directory_iterator(st.root() / "objects"))
    CHECK(e.path().filename().string().find(".tmp.") == std::string::npos);
}

static json Core(const std::string& in) {
  return {{"layer", "shader_ps"}, {"game", "bo2"}, {"source", "selftest"},
          {"inputs", {{"ucode", in}, {"consts", std::string(64, 'c')}}},
          {"params", {{"count", 64}, {"interp_mask", 3}}},
          {"outputs", {{"oC0", {{"elem", "f32"}, {"shape", {64, 4}}, {"compare", "bits"}}},
                       {"fetch_trace", {{"elem", "u8"}, {"shape", {8}}, {"compare", "none"}}}}}};
}

static void TestManifest(Store& st) {
  std::string h1 = st.PutObject("aaaa", 4), h2 = st.PutObject("bbbb", 4), h3 = st.PutObject("trace", 5);
  std::string in = std::string(64, '1');
  json core = Core(in);

  // Registry: rule r1 v2.
  CHECK(st.RegisterRule("r1", 2, "assumed", true));
  CHECK(!st.RegisterRule("r1", 1, "x", false));  // never lowers
  CHECK(st.BumpRule("zzz") == -1);

  std::string id;
  json a = {{"version", "abc123"}, {"rules", {{"r1", 2}}}, {"outputs", {{"oC0", h1}, {"fetch_trace", h3}}}};
  CHECK(st.WriteCaseFromReferee(core, "refxenos", a, &id));
  json m;
  CHECK(st.ReadCase("shader_ps", id, &m));
  CHECK_EQ(m["status"].get<std::string>(), "single");
  CHECK(m["golden"] == json({{"oC0", h1}}));  // compare:none excluded
  CHECK(m["schema"] == 1 && m["id"] == id && m["game"] == "bo2");
  CHECK(fs::exists(st.root() / "cases/shader_ps" / (id + ".json")));

  // Second referee agrees (extra producer-specific names ignored for agreement).
  json b = {{"version", "def456"}, {"rules", {{"r1", 2}}}, {"outputs", {{"oC0", h1}}}};
  std::string id2;
  CHECK(st.WriteCaseFromReferee(core, "xenia_fsi", b, &id2));
  CHECK_EQ(id2, id);
  CHECK(st.ReadCase("shader_ps", id, &m));
  CHECK_EQ(m["status"].get<std::string>(), "agreed");
  CHECK(m["golden"] == json({{"oC0", h1}}));
  CHECK(m["referees"].size() == 2);
  CHECK(m["referees"]["refxenos"] == a);  // untouched

  // Referee re-runs with a different result: disputed, moves to disputes/.
  json b2 = {{"version", "def457"}, {"rules", {{"r1", 2}}}, {"outputs", {{"oC0", h2}}}};
  CHECK(st.WriteCaseFromReferee(core, "xenia_fsi", b2, &id2));
  CHECK(!fs::exists(st.root() / "cases/shader_ps" / (id + ".json")));
  CHECK(fs::exists(st.root() / "disputes/shader_ps" / (id + ".json")));
  CHECK(st.ReadCase("shader_ps", id, &m));
  CHECK_EQ(m["status"].get<std::string>(), "disputed");
  CHECK(!m.contains("golden"));
  CHECK(m["referees"]["refxenos"] == a);

  // Staleness: bump r1 to 3 -> both entries stale (rule 2 < 3).
  CHECK(st.BumpRule("r1") == 3);
  json reg;
  CHECK(st.LoadRules(&reg));
  CHECK(reg["r1"]["version"] == 3 && reg["r1"]["assumed"] == true);
  CHECK(st.IsEntryStale(a, reg));
  CHECK(st.IsEntryStale(b2));
  CHECK(!st.IsEntryStale({{"outputs", json::object()}}));
  CHECK(!st.IsEntryStale({{"rules", {{"unregistered", 1}}}}));
  CHECK(st.StaleRules(a, reg) == std::vector<std::string>{"r1"});

  // refxenos refreshes (rules r1=3): xenia_fsi's stale entry is ignored, so
  // the case is single again and returns to cases/.
  json a3 = {{"version", "abc999"}, {"rules", {{"r1", 3}}}, {"outputs", {{"oC0", h2}}}};
  CHECK(st.WriteCaseFromReferee(core, "refxenos", a3, &id));
  CHECK(st.ReadCase("shader_ps", id, &m));
  CHECK_EQ(m["status"].get<std::string>(), "single");
  CHECK(m["golden"] == json({{"oC0", h2}}));
  CHECK(fs::exists(st.root() / "cases/shader_ps" / (id + ".json")));
  CHECK(!fs::exists(st.root() / "disputes/shader_ps" / (id + ".json")));

  // xenia_fsi refreshes and now agrees with refxenos.
  json b3 = {{"version", "def900"}, {"rules", {{"r1", 3}}}, {"outputs", {{"oC0", h2}}}};
  CHECK(st.WriteCaseFromReferee(core, "xenia_fsi", b3, &id));
  CHECK(st.ReadCase("shader_ps", id, &m));
  CHECK_EQ(m["status"].get<std::string>(), "agreed");
  CHECK(m["golden"] == json({{"oC0", h2}}));
}

// Two processes merging the same case concurrently; both entries must survive.
static void TestConcurrent(const fs::path& root) {
  Store st(root);
  json core = Core(std::string(64, '9'));
  std::string h = st.PutObject("zz", 2);
  const int kProcs = 8;
  for (int i = 0; i < kProcs; ++i) {
    pid_t pid = fork();
    if (pid == 0) {
      Store s(root);
      std::string id;
      bool ok = true;
      for (int k = 0; k < 20; ++k) {
        json e = {{"version", "v" + std::to_string(k)}, {"outputs", {{"oC0", h}}}};
        ok &= s.WriteCaseFromReferee(core, "ref" + std::to_string(i), e, &id);
      }
      _exit(ok ? 0 : 1);
    }
  }
  int bad = 0, status;
  while (wait(&status) > 0) bad += (!WIFEXITED(status) || WEXITSTATUS(status) != 0);
  CHECK(bad == 0);
  json m;
  CHECK(st.ReadCase("shader_ps", st.CaseId(core), &m));
  CHECK(m["referees"].size() == size_t(kProcs));
  CHECK_EQ(m["status"].get<std::string>(), "agreed");
}

int main() {
  fs::path root = kScratch / "store";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(kScratch);
  Store st(root);
  TestSha();
  TestCanonical(st);
  TestObjects(st);
  TestManifest(st);
  TestConcurrent(root);
  std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
