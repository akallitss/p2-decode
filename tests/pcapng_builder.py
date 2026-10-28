"""Build small synthetic SRS/VMM3a pcapng files with known contents."""

import struct


def gray(n):
    return n ^ (n >> 1)


def hit_word(vmm, ch, adc, bcid, tdc, offset=0, ot=0):
    """6 bytes of one VMM hit, as the FEC sends it (bcid given in binary)."""
    d1 = ((offset & 0x1F) << 27) | ((vmm & 0x1F) << 22) | ((adc & 0x3FF) << 12) \
        | (gray(bcid) & 0xFFF)
    d2 = 0x8000 | ((ot & 1) << 14) | ((ch & 0x3F) << 8) | (tdc & 0xFF)
    return struct.pack(">IH", d1, d2)


def marker_word(vmm, timestamp42):
    """6 bytes of one SRS timestamp marker for `vmm`."""
    d1 = (timestamp42 >> 10) & 0xFFFFFFFF
    d2 = ((vmm & 0x1F) << 10) | (timestamp42 & 0x3FF)
    return struct.pack(">IH", d1, d2)


def srs_payload(frame_counter, fec, words):
    """UDP payload: 16-byte SRS header followed by the 6-byte words."""
    hdr = struct.pack(">I", frame_counter) + b"VM3" + bytes([(fec & 0x0F) << 4])
    hdr += b"\x00" * 8            # UDP timestamp, offset overflow
    return hdr + b"".join(words)


def _udp_frame(payload, src_ip):
    a, b, c, d = (int(x) for x in src_ip.split("."))
    eth = b"\xff" * 6 + b"\x02\x00\x00\x00\x00\x01" + b"\x08\x00"
    udp = struct.pack(">HHHH", 6006, 6006, 8 + len(payload), 0)
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp) + len(payload),
                     0, 0, 64, 17, 0, bytes([a, b, c, d]), bytes([10, 0, 0, 3]))
    return eth + ip + udp + payload


def _block(btype, body):
    body += b"\x00" * (-len(body) % 4)
    n = 12 + len(body)
    return struct.pack("<II", btype, n) + body + struct.pack("<I", n)


def write_pcapng(path, payloads, src_ip="10.0.0.2", tsresol=None):
    """Write one Ethernet/IPv4/UDP packet per payload, packet i stamped i ticks.

    tsresol: the IDB's if_tsresol option (None = absent = microseconds).
    """
    idb = struct.pack("<HHI", 1, 0, 0)
    if tsresol is not None:
        idb += struct.pack("<HHB3x", 9, 1, tsresol) + struct.pack("<HH", 0, 0)
    out = [_block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1)),
           _block(0x00000001, idb)]
    for i, p in enumerate(payloads):
        frame = _udp_frame(p, src_ip)
        out.append(_block(0x00000006, struct.pack("<IIIII", 0, 0, i,
                                                  len(frame), len(frame)) + frame))
    with open(path, "wb") as f:
        f.write(b"".join(out))
    return path
