// refxenos build shim. The SDK's <rex/memory.h> pulls in the whole guest
// memory system (arenas, mapped memory, ring buffers, kernel xmemory). The
// graphics headers refxenos uses (xenos.h, registers.h, register_file.h,
// ucode.h) only need the helpers in <rex/memory/utils.h> (fourcc, Reinterpret,
// load/store with byte swap), so the standalone refxenos build puts this
// directory first on the include path. Not used by any SDK target.
#pragma once
#include <rex/memory/utils.h>
