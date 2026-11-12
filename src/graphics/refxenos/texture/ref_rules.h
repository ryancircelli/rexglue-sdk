// Rule registry for the refxenos texture reference. Each rule the decoder
// applies has an id, a version (bump when the rule's behaviour changes, so
// goldens produced by the old version go stale), a status and its source.
#pragma once

#include <cstddef>
#include <cstdint>

namespace refxenos::tex {

enum class RuleStatus {
  kDocumented,  // stated by the XDK documentation
  kXenia,       // taken from Xenia / ReXGlue SDK code (reverse engineered)
  kAssumed,     // no source found; a real-hardware question
};

struct RuleInfo {
  const char* id;
  uint32_t version;
  RuleStatus status;
  const char* summary;
  const char* source;
  // Real-Xbox test that settles the rule (required for kAssumed rules):
  // what to upload, how to fetch it, and what to read back.
  const char* hw_test = nullptr;
};

const RuleInfo* GetRules(size_t* count);
const RuleInfo* FindRule(const char* id);
const char* RuleStatusName(RuleStatus status);

}  // namespace refxenos::tex
