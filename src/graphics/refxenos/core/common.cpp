#include "common.h"

#include <cstdio>

namespace refxenos {

RuleRegistry& RuleRegistry::Get() {
  static RuleRegistry registry;
  return registry;
}

RuleUsage& RuleUsage::Current() {
  thread_local RuleUsage usage;
  return usage;
}

Probe& Probe::Get() {
  static Probe probe;
  return probe;
}

void Probe::Log(int32_t x, int32_t y, const std::string& stage, const std::string& text) {
  char prefix[64];
  std::snprintf(prefix, sizeof(prefix), "px(%d,%d) %s: ", x, y, stage.c_str());
  lines_.push_back(prefix + text);
}

void Probe::LogVertex(uint32_t vertex, const std::string& stage, const std::string& text) {
  char prefix[64];
  std::snprintf(prefix, sizeof(prefix), "vtx%u %s: ", vertex, stage.c_str());
  lines_.push_back(prefix + text);
}

std::string FormatFloat4(const float* v) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "(%.9g, %.9g, %.9g, %.9g) [%08x %08x %08x %08x]", v[0], v[1],
                v[2], v[3], AsU32(v[0]), AsU32(v[1]), AsU32(v[2]), AsU32(v[3]));
  return buf;
}

}  // namespace refxenos
