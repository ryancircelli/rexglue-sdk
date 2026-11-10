#!/usr/bin/env python3
"""Convert a rexglue .xtr GPU trace into one Xenia Canary's trace tools read.

The two formats share trace_protocol.h (version 1, same commands and struct
layouts). The one difference is that rexglue may zlib-compress memory blocks
(MemoryEncodingFormat 2, a rexglue extension). Xenia only reads kNone (0) and
kSnappy (1), so every zlib block is rewritten as kNone.

Everything else is copied byte for byte:
- PrimaryBufferStart/IndirectBufferStart with count 0 (rexglue does not copy
  the buffer contents there; Xenia skips `count * 4` bytes, i.e. nothing).
- Indirect buffer packets are recorded as an outer INDIRECT_BUFFER
  PacketStart, nested PacketStart/End pairs for its contents, then the outer
  PacketEnd. Xenia's TracePlayer keeps a single pending packet, so the inner
  starts replace the outer one and the outer PacketEnd executes nothing: each
  packet runs exactly once, as in rexgpu-replay.
- WAIT_REG_MEM: rexglue records the polled dword after the wait matched, so
  Xenia's polling loop sees a matching value on its first read.

  xtr_to_xenia.py IN.xtr OUT.xtr
"""

import mmap
import struct
import sys
import zlib

NAMES = ["PrimaryBufferStart", "PrimaryBufferEnd", "IndirectBufferStart", "IndirectBufferEnd",
         "PacketStart", "PacketEnd", "MemoryRead", "MemoryWrite", "EdramSnapshot", "Event",
         "Registers", "GammaRamp"]


def convert(src, dst):
    with open(src, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    out = open(dst, "wb")
    version, = struct.unpack_from("<I", data, 0)
    if version != 1:
        raise SystemExit(f"unsupported trace version {version}")
    out.write(data[:48])
    off = 48
    n = len(data)
    stats = {"zlib_blocks": 0, "bytes_in": n, "packets": 0, "frames": 0,
             "truncated": False}
    try:
        _convert_records(data, off, n, out, stats)
    except (struct.error, zlib.error, _Truncated):
        stats["truncated"] = True
    # Captures cut off at game exit end mid-record. Keep whole frames only:
    # Xenia's reader does not bounds-check the last command.
    if stats["truncated"] and stats.get("last_swap_end"):
        out.truncate(stats["last_swap_end"])
        out.seek(stats["last_swap_end"])
    stats["bytes_out"] = out.tell()
    out.close()
    return stats


class _Truncated(Exception):
    pass


def _convert_records(data, off, n, out, stats):
    def need(end):
        if end > n:
            raise _Truncated()

    while off < n:
        need(off + 4)
        t, = struct.unpack_from("<I", data, off)
        if t in (0, 2):
            base, count = struct.unpack_from("<II", data, off + 4)
            end = off + 12 + count * 4
            need(end)
            out.write(data[off:end])
            off = end
        elif t in (1, 3, 5):
            out.write(data[off:off + 4])
            off += 4
        elif t == 4:
            base, count = struct.unpack_from("<II", data, off + 4)
            end = off + 12 + count * 4
            need(end)
            out.write(data[off:end])
            off = end
            stats["packets"] += 1
        elif t in (6, 7):
            base, enc, elen, dlen = struct.unpack_from("<IIII", data, off + 4)
            need(off + 20 + elen)
            payload = data[off + 20:off + 20 + elen]
            if enc == 2:
                raw = zlib.decompress(payload)
                if len(raw) != dlen:
                    raise SystemExit(f"zlib block at {off}: {len(raw)} bytes, expected {dlen}")
                out.write(struct.pack("<IIIII", t, base, 0, dlen, dlen))
                out.write(raw)
                stats["zlib_blocks"] += 1
            else:
                out.write(data[off:off + 20 + elen])
            off += 20 + elen
        elif t == 8:
            enc, elen, dlen = struct.unpack_from("<III", data, off + 4)
            need(off + 16 + elen)
            payload = data[off + 16:off + 16 + elen]
            if enc == 2:
                raw = zlib.decompress(payload)
                out.write(struct.pack("<IIII", t, 0, len(raw), len(raw)))
                out.write(raw)
                stats["zlib_blocks"] += 1
            else:
                out.write(data[off:off + 16 + elen])
            off += 16 + elen
        elif t == 9:
            need(off + 8)
            out.write(data[off:off + 8])
            ev, = struct.unpack_from("<I", data, off + 4)
            off += 8
            if ev == 0:  # EventCommand::Type::kSwap
                stats["frames"] += 1
                stats["last_swap_end"] = out.tell()
        elif t == 10:
            need(off + 24)
            first, count, cb, enc, elen = struct.unpack_from("<IIBxxxII", data, off + 4)
            need(off + 24 + elen)
            if enc == 2:
                raw = zlib.decompress(data[off + 24:off + 24 + elen])
                out.write(data[off:off + 16])
                out.write(struct.pack("<II", 0, len(raw)))
                out.write(raw)
                stats["zlib_blocks"] += 1
            else:
                out.write(data[off:off + 24 + elen])
            off += 24 + elen
        elif t == 11:
            need(off + 16)
            rw, enc, elen = struct.unpack_from("<BxxxII", data, off + 4)
            need(off + 16 + elen)
            if enc == 2:
                raw = zlib.decompress(data[off + 16:off + 16 + elen])
                out.write(data[off:off + 8])
                out.write(struct.pack("<II", 0, len(raw)))
                out.write(raw)
                stats["zlib_blocks"] += 1
            else:
                out.write(data[off:off + 16 + elen])
            off += 16 + elen
        else:
            raise SystemExit(f"unknown command {t} at offset {off}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    print(convert(sys.argv[1], sys.argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
