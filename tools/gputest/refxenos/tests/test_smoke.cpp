#include <cstdio>
#include "core/common.h"
#include <rex/graphics/format/ucode.h>
int main() {
  refxenos::RegisterFile* rf = new refxenos::RegisterFile;
  (void)rf;
  std::printf("rules registered: %zu\n", refxenos::RuleRegistry::Get().rules().size());
  return 0;
}
