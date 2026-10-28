"""pcapng container and link layer: locate every UDP payload in a capture.

Knows nothing about SRS or VMM: it returns where each UDP payload starts, how
long it is and which IPv4 address sent it, vectorised over packets.
"""

import os
import struct

import numpy as np

# pcapng header words held in RAM at a time (64 MB); module-level so tests can
# shrink it to exercise the slab-reload path
_SLAB_WORDS = 1 << 24

# pcapng block types
_BT_SHB = 0x0A0D0D0A
_BT_IDB = 0x00000001
_BT_EPB = 0x00000006
_BT_SPB = 0x00000003

_BOM = 0x1A2B3C4D
_LINKTYPE_ETHERNET = 1


def _ts_unit_ns(tsresol):
    """Nanoseconds per timestamp tick for a pcapng if_tsresol byte."""
    if tsresol & 0x80:
        return 1e9 / (1 << (tsresol & 0x7F))
    return 10.0 ** (9 - tsresol)


def _idb_tsresol(u8, pos, blen, endian):
    """if_tsresol of the IDB at `pos` (default 6, i.e. microseconds)."""
    o, end = pos + 16, pos + blen - 4
    while o + 4 <= end:
        code, ln = struct.unpack_from(endian + "HH", bytes(u8[o:o + 4]))
        if code == 0:
            break
        if code == 9 and ln >= 1:
            return int(u8[o + 4])
        o += 4 + ln + (-ln % 4)
    return 6


def walk_blocks(path, max_packets=None, times=False):
    """Return (u8, pkt_off, pkt_caplen, linktypes), plus pkt_time_ns if `times`.

    pkt_time_ns is each packet's capture time in ns since the Unix epoch,
    float64 (resolution ~0.25 us at today's epoch), NaN for Simple Packet Blocks,
    which carry none. The interface's if_tsresol is honoured.

    Walks the pcapng block chain to find every Enhanced/Simple Packet Block.
    Only block headers are read in Python -- roughly one iteration per packet,
    which is ~1e3 for a 45 s capture and ~2e5 for a 2 GB file.

    The file is memory-mapped rather than read: a 2 GB capture must not be
    pulled into a 7 GB machine that kills QA at 80 % RAM. Block headers are
    served from a 64 MB in-RAM slab rather than read from the memmap one scalar
    at a time -- memmap.__getitem__ costs ~75 us per scalar, which on a 2 GB
    file turns the walk alone into ~35 s.

    `max_packets` stops the walk early; without it a --max-packets=3000 run on
    a 2 GB capture still pays for chasing all 230k blocks.
    """
    size = os.path.getsize(path)
    if size < 12:
        raise ValueError(f"{path}: too small to be a pcapng")

    u8 = np.memmap(path, dtype=np.uint8, mode="r")

    head = bytes(u8[:12])
    btype0 = struct.unpack_from("<I", head, 0)[0]
    if btype0 != _BT_SHB:
        # A classic .pcap starts with 0xa1b2c3d4 / 0xd4c3b2a1 instead.
        if struct.unpack_from("<I", head, 0)[0] in (0xA1B2C3D4, 0xD4C3B2A1):
            raise ValueError(
                f"{path}: this is a classic .pcap, not a .pcapng. "
                "Only pcapng is produced by the DAQ's dumpcap; convert with editcap."
            )
        raise ValueError(f"{path}: not a pcapng (first block type 0x{btype0:08x})")

    bom = struct.unpack_from("<I", head, 8)[0]
    endian = "<" if bom == _BOM else ">"
    u32 = np.memmap(path, dtype=(endian + "u4"), mode="r",
                    shape=(size // 4,))

    pkt_off = []
    pkt_len = []
    linktypes = []
    pkt_time = []
    ts_units = []            # ns per tick, per interface, in IDB order

    n_words = size // 4
    slab_words = _SLAB_WORDS
    slab = None
    slab_lo = slab_hi = -1

    pos = 0
    n = size
    while pos + 12 <= n:
        w = pos >> 2
        # Reload when w falls outside the slab, or when there is not enough
        # headroom left for a block header -- unless the slab already reaches
        # EOF, in which case no reload can help and we must not loop forever.
        if not (slab_lo <= w < slab_hi) or (w + 8 >= slab_hi < n_words):
            slab_lo = w
            slab_hi = min(w + slab_words, n_words)
            if slab_hi - slab_lo < 2:
                break
            slab = np.array(u32[slab_lo:slab_hi])   # one sequential copy
        i = w - slab_lo
        avail = slab_hi - slab_lo          # words readable at slab[i + k]
        btype = int(slab[i])
        blen = int(slab[i + 1])
        if blen < 12 or blen % 4 or pos + blen > n:
            # Truncated tail: dumpcap was mid-write, or the file was cut.
            break
        if btype == _BT_EPB:
            #  interface_id, ts_hi, ts_lo, caplen, origlen, data...
            if blen < 32 or i + 5 >= avail:
                break
            pkt_off.append(pos + 28)
            pkt_len.append(int(slab[i + 5]))
            if times:
                iface = int(slab[i + 2])
                unit = ts_units[iface] if iface < len(ts_units) else 1e3
                pkt_time.append(((int(slab[i + 3]) << 32) | int(slab[i + 4])) * unit)
        elif btype == _BT_SPB:
            # Simple Packet Block has no caplen; the block length bounds it.
            if blen < 16 or i + 2 >= avail:
                break
            pkt_off.append(pos + 12)
            pkt_len.append(min(int(slab[i + 2]), blen - 16))
            if times:
                pkt_time.append(np.nan)
        elif btype == _BT_IDB:
            if blen < 20 or i + 2 >= avail:
                break
            linktypes.append(int(slab[i + 2]) & 0xFFFF)
            if times:
                ts_units.append(_ts_unit_ns(_idb_tsresol(u8, pos, blen, endian)))
        elif btype == _BT_SHB:
            if blen < 28 or i + 2 >= avail:
                break
            if int(slab[i + 2]) not in (_BOM, 0x4D3C2B1A):
                break
        pos += blen
        if max_packets is not None and len(pkt_off) >= max_packets:
            break

    del slab
    out = (u8,
           np.asarray(pkt_off, dtype=np.int64),
           np.asarray(pkt_len, dtype=np.int64),
           linktypes)
    if times:
        out += (np.asarray(pkt_time, dtype=np.float64),)
    return out


def read_packets(path, max_packets=None, times=False):
    """Locate every UDP payload in a pcapng, vectorised over packets.

    Returns (u8, payload_start, payload_len, src_ip_u32, n_packets), plus the
    UDP packets' capture times in ns (see walk_blocks) if `times`.
    `n_packets` counts every captured packet, matching the reference parser's
    `pkt_count` (which drives --max-packets), not just the UDP ones.
    """
    walked = walk_blocks(path, max_packets=max_packets, times=times)
    u8, off, caplen, linktypes = walked[:4]
    t_ns = walked[4] if times else None

    if linktypes and any(lt != _LINKTYPE_ETHERNET for lt in linktypes):
        raise ValueError(
            f"{path}: link type(s) {sorted(set(linktypes))} -- only Ethernet (1) "
            "is supported; the DAQ captures Ethernet."
        )

    n_packets = len(off)
    if max_packets is not None and n_packets > max_packets:
        off = off[:max_packets]
        caplen = caplen[:max_packets]
        if times:
            t_ns = t_ns[:max_packets]
        n_packets = len(off)

    if n_packets == 0:
        empty = np.zeros(0, dtype=np.int64)
        out = (u8, empty, empty, np.zeros(0, dtype=np.uint32), 0)
        return out + ((np.zeros(0),) if times else ())

    # --- Ethernet, with any number of VLAN tags (802.1Q / 802.1ad) ---
    ok = caplen >= 14
    ethertype = np.zeros(n_packets, dtype=np.uint32)
    l3 = off + 14
    idx = np.flatnonzero(ok)
    ethertype[idx] = (u8[off[idx] + 12].astype(np.uint32) << 8 |
                      u8[off[idx] + 13].astype(np.uint32))
    for _ in range(3):  # QinQ nests at most a couple deep in practice
        vlan = ok & ((ethertype == 0x8100) | (ethertype == 0x88A8) |
                     (ethertype == 0x9100))
        if not vlan.any():
            break
        vi = np.flatnonzero(vlan)
        l3[vi] += 4
        good = vi[(l3[vi] + 2) <= (off[vi] + caplen[vi])]
        ethertype[vi] = 0
        ethertype[good] = (u8[l3[good] - 2].astype(np.uint32) << 8 |
                           u8[l3[good] - 1].astype(np.uint32))

    # --- IPv4 + UDP ---
    ok &= ethertype == 0x0800
    ok &= (l3 + 20) <= (off + caplen)
    idx = np.flatnonzero(ok)

    ihl = np.zeros(n_packets, dtype=np.int64)
    proto = np.zeros(n_packets, dtype=np.uint8)
    src = np.zeros(n_packets, dtype=np.uint32)
    if len(idx):
        vers_ihl = u8[l3[idx]]
        ihl[idx] = (vers_ihl & 0x0F).astype(np.int64) * 4
        proto[idx] = u8[l3[idx] + 9]
        src[idx] = (u8[l3[idx] + 12].astype(np.uint32) << 24 |
                    u8[l3[idx] + 13].astype(np.uint32) << 16 |
                    u8[l3[idx] + 14].astype(np.uint32) << 8 |
                    u8[l3[idx] + 15].astype(np.uint32))
        ok[idx] &= ((vers_ihl >> 4) == 4) & (ihl[idx] >= 20) & (proto[idx] == 17)

    l4 = l3 + ihl
    pstart = l4 + 8
    # UDP length field bounds the payload; fall back to the capture length.
    plen = np.zeros(n_packets, dtype=np.int64)
    idx = np.flatnonzero(ok & ((l4 + 8) <= (off + caplen)))
    if len(idx):
        udp_len = (u8[l4[idx] + 4].astype(np.int64) << 8 |
                   u8[l4[idx] + 5].astype(np.int64))
        avail = off[idx] + caplen[idx] - pstart[idx]
        plen[idx] = np.minimum(np.maximum(udp_len - 8, 0), avail)
    keep = plen > 0

    out = (u8, pstart[keep], plen[keep], src[keep], n_packets)
    return out + ((t_ns[keep],) if times else ())


def ip_str(v):
    return f"{(v >> 24) & 0xFF}.{(v >> 16) & 0xFF}.{(v >> 8) & 0xFF}.{v & 0xFF}"


def ip_u32(s):
    a, b, c, d = (int(x) for x in s.split("."))
    return np.uint32((a << 24) | (b << 16) | (c << 8) | d)
