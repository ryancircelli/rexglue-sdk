// refxenos command line.
//
//   refxenos --rules-md              assumed-rules table (research/reference-renderer.md)
//   refxenos --rules-json            every rule with version, source, assumed flag, hw test
//   refxenos replay TRACE -o DIR     render a capture, write the rexgpu-replay layout
//            [--frames N] [--draws FIRST:COUNT] [--full] [--label L]
//            [--probe X,Y[,DRAW]]... [--probe-out FILE]
#include <cstdio>
#include <cstring>
#include <string>

#include "output/replay_writer.h"

namespace refxenos {
int RunReplay(int argc, char** argv);  // pipeline/replay.cpp (weak until wired)
}

int main(int argc, char** argv) {
  if (argc >= 2 && !std::strcmp(argv[1], "--rules-md")) {
    std::fputs(refxenos::RulesMarkdown().c_str(), stdout);
    return 0;
  }
  if (argc >= 2 && !std::strcmp(argv[1], "--rules-json")) {
    std::fputs(refxenos::RulesJson().c_str(), stdout);
    return 0;
  }
  if (argc >= 2 && !std::strcmp(argv[1], "replay")) {
    return refxenos::RunReplay(argc - 1, argv + 1);
  }
  std::fprintf(stderr,
               "usage: refxenos --rules-md | --rules-json | replay TRACE -o DIR [options]\n");
  return 2;
}
