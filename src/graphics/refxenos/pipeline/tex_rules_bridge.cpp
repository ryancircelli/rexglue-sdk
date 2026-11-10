// Bridges the texture library's rule registry (texture/ref_rules.cc, owned by
// the texconf branch refxenos-tex) into refxenos's RuleRegistry, so texture
// rules show up in `refxenos --rules-md`, rules.json and per-draw rule lists.
//
// tex::RuleInfo has no separate hardware-test field yet; for assumed rules the
// summary text stands in until the library adds one.

#include "pipeline/tex_rules_bridge.h"

#include <deque>
#include <map>
#include <string>

#include "texture/ref_rules.h"

namespace refxenos {
namespace {

struct TexRuleTable {
  std::deque<std::string> strings;  // stable storage for c-strings
  std::deque<Rule> rules;
  std::map<std::string, const Rule*> by_id;

  TexRuleTable() {
    size_t count = 0;
    const tex::RuleInfo* infos = tex::GetRules(&count);
    for (size_t i = 0; i < count; ++i) {
      const tex::RuleInfo& info = infos[i];
      const char* id = strings.emplace_back(std::string("tex.") + info.id).c_str();
      const char* source =
          strings
              .emplace_back(std::string(tex::RuleStatusName(info.status)) + ": " +
                            (info.source ? info.source : "") + " | " +
                            (info.summary ? info.summary : ""))
              .c_str();
      bool assumed = info.status == tex::RuleStatus::kAssumed;
      const char* hw = assumed ? (info.summary ? info.summary : "") : "";
      rules.push_back(Rule{id, static_cast<int>(info.version), source, assumed, hw});
      RuleRegistry::Get().Register(&rules.back());
      by_id[info.id] = &rules.back();
    }
  }
};

TexRuleTable& Table() {
  static TexRuleTable table;
  return table;
}

// Register at static-init time so --rules-md lists texture rules without a
// texture ever being sampled.
[[maybe_unused]] const bool kRegistered = (Table(), true);

}  // namespace

void UseTexRules(const std::vector<std::string>& ids, DrawStats* stats) {
  auto& table = Table();
  for (const std::string& id : ids) {
    auto it = table.by_id.find(id);
    if (it != table.by_id.end()) {
      RuleUsage::Current().Use(it->second);
    } else if (stats) {
      stats->notes.push_back("texture rule not in registry: " + id);
    }
  }
}

}  // namespace refxenos
