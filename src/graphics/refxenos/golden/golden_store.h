// Shared golden store writer/reader for the BO2 reference-renderer project.
// Implements /Users/ryanc/dev/bo2-mac-port/research/golden-store.md (v1.1).
// Dependencies: C++20 standard library + nlohmann/json only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace refxenos::golden {

// FIPS 180-2 SHA-256; returns 64 lowercase hex chars.
std::string Sha256Hex(const void* data, size_t size);

// Canonical JSON (spec "Layout": "sorted keys, no spaces"). Byte-identical to
// Python json.dumps(obj, sort_keys=True, separators=(",", ":")) with the
// default ensure_ascii=True (non-ASCII -> \uXXXX lowercase, surrogate pairs),
// floats via Python repr (shortest round-trip, "1e-05", "100000.0").
std::string CanonicalJson(const nlohmann::json& j);

// Registry golden/rules.json (spec "Rule versions"):
// {"<rule_id>": {"version": N, "source": "...", "assumed": bool}}
class Store {
 public:
  explicit Store(std::filesystem::path root);

  const std::filesystem::path& root() const { return root_; }

  // ---- objects/<h[0:2]>/<h> (immutable, temp file + rename, never rewritten)
  std::string PutObject(const void* data, size_t size);
  bool GetObject(const std::string& hash, std::vector<uint8_t>* out) const;
  bool HasObject(const std::string& hash) const;
  std::filesystem::path ObjectPath(const std::string& hash) const;

  // ---- cases
  // sha256 of canonical JSON of {"layer","inputs","params"} taken from the
  // given object (other keys ignored; absent "params" counts as {}).
  std::string CaseId(const nlohmann::json& layer_inputs_params) const;

  // Reads cases/<layer>/<id>.json, else disputes/<layer>/<id>.json.
  bool ReadCase(const std::string& layer, const std::string& id,
                nlohmann::json* out) const;

  // Create or merge a case. case_core: layer, inputs, params, source, game,
  // outputs (the outputs spec: name -> {elem, shape, compare}). Only
  // `referee`'s entry in "referees" is touched. Recomputes status/golden and
  // moves the file between cases/ and disputes/ as needed. Returns false on
  // I/O error or malformed input (error text in *error if given).
  bool WriteCaseFromReferee(const nlohmann::json& case_core,
                            const std::string& referee,
                            const nlohmann::json& referee_entry,
                            std::string* case_id_out,
                            std::string* error = nullptr);

  // ---- rules registry
  bool LoadRules(nlohmann::json* out) const;  // {} if file absent
  // A referee entry is stale if any listed rule's version is below the
  // registry's. Rules absent from the registry are not stale.
  bool IsEntryStale(const nlohmann::json& referee_entry,
                    const nlohmann::json& registry) const;
  bool IsEntryStale(const nlohmann::json& referee_entry) const;
  // Stale rule ids of the entry (for reporting).
  std::vector<std::string> StaleRules(const nlohmann::json& referee_entry,
                                      const nlohmann::json& registry) const;
  // Register a rule (version only ever goes up; existing source/assumed are
  // replaced by the given values when the version is raised or the rule is new).
  bool RegisterRule(const std::string& rule_id, int64_t version,
                    const std::string& source, bool assumed);
  // Increment the registry version of an existing rule; returns the new
  // version, or -1 on failure / unknown rule.
  int64_t BumpRule(const std::string& rule_id);

 private:
  std::filesystem::path CasePath(const std::string& layer,
                                 const std::string& id) const;
  std::filesystem::path DisputePath(const std::string& layer,
                                    const std::string& id) const;
  std::filesystem::path root_;
};

// Recompute "status" and "golden" of a manifest from its referees, ignoring
// stale entries (spec "Case manifest" + "Rule versions"). Exposed for tests.
void RecomputeStatus(nlohmann::json* manifest, const nlohmann::json& registry);

}  // namespace refxenos::golden
