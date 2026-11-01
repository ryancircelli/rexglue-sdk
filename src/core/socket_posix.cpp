#include <rex/net/socket.h>
#include <rex/platform.h>

static_assert(REX_PLATFORM_LINUX || REX_PLATFORM_MAC, "This file is POSIX-only");

#include <sys/ioctl.h>
#include <unistd.h>

namespace rex::net {

int socket_close(SocketHandle handle) {
  return close(static_cast<int>(handle));
}

int socket_ioctl(SocketHandle handle, uint32_t cmd, uint8_t* arg) {
#if defined(__linux__)
  // The guest passes Winsock's command numbers, which are the BSD ones (and so macOS's). Linux
  // numbers its own: without this FIONBIO fails and a title stops at start-up ("Failed to set
  // socket to nonblocking").
  if (cmd == 0x8004667Eu) {  // _IOW('f', 126, u_long)
    cmd = FIONBIO;
  } else if (cmd == 0x4004667Fu) {  // _IOR('f', 127, u_long)
    cmd = FIONREAD;
  }
#endif
  return ioctl(static_cast<int>(handle), cmd, arg);
}

}  // namespace rex::net
