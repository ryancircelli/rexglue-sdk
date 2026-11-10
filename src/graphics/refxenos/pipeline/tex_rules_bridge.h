#pragma once

#include <string>
#include <vector>

#include "core/common.h"

namespace refxenos {

// Marks the texture-library rules named in tex::Diag::rules_used as used by
// the current unit of work. Texture rule ids appear as "tex.<id>".
void UseTexRules(const std::vector<std::string>& ids, DrawStats* stats);

}  // namespace refxenos
