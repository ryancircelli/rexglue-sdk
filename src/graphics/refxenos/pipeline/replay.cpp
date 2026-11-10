// refxenos replay driver: trace -> command processor -> per-draw pipeline -> writer.
// Wired once the cp/, shader/, raster and rop modules land (see
// research/reference-renderer.md).
#include <cstdio>

namespace refxenos {

int RunReplay(int, char**) {
  std::fprintf(stderr, "refxenos replay: pipeline not wired yet\n");
  return 1;
}

}  // namespace refxenos
