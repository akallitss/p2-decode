"""Decoding a capture while dumpcap is still writing it.

    sd = StreamDecoder("capture.pcapng")
    while running:
        cols, info = sd.poll()        # hits that arrived since the last poll
        ...

PcapngFollower returns only complete pcapng blocks appended since its last
call; a block dumpcap has half written stays for the next one. StreamDecoder
feeds those packets to the same WordDecoder the whole-file path uses, keeping
the marker state between polls, so the hits of all polls together are the
hits of decoding the finished file (tests/test_follow.py checks this byte
split by byte split).

One difference from decode(): with src_ips=None, decode() keeps the FEC IPs
seen in the first 500 packets; the stream keeps every source that sends VM3
payloads. For one FEC, as at SPS and in the lab, the two agree.
"""

import os
import struct

import numpy as np

from .pcapng import (_BOM, _BT_EPB, _BT_IDB, _BT_SHB, _BT_SPB, _ts_unit_ns,
                     check_linktypes, ip_str, ip_u32, udp_payloads)
from .srs import MAX_WORDS, WordDecoder, packet_header, select_vm3

# bytes read per poll at most, so a poll after a long pause stays bounded
MAX_READ = 256 * 1024 * 1024


class PcapngFollower:
    """Packets of a growing pcapng, as they are appended."""

    def __init__(self, path, max_read=MAX_READ):
        self.path = str(path)
        self.max_read = max_read
        self.pos = 0                    # bytes consumed (whole blocks only)
        self.endian = "<"
        self.ts_units = []              # ns per tick, per interface of the section
        self.linktypes = []
        self.n_packets = 0
        self.skipped_bytes = 0
        self.skipped_packets = 0

    def backlog(self):
        """Bytes written to the file but not read yet."""
        try:
            return max(0, os.path.getsize(self.path) - self.pos)
        except OSError:
            return 0

    def skip_to_tail(self, keep_bytes):
        """Jump ahead so at most ~keep_bytes remain unread. Returns bytes skipped.

        Walks block headers only (8 bytes per block), so it costs a seek per
        block, not a read of the data. Section and interface blocks met on the
        way are still parsed, so timestamps and byte order stay right after the
        jump. Packets skipped are counted in skipped_packets / skipped_bytes.
        """
        size = os.path.getsize(self.path)
        target = size - keep_bytes
        if target <= self.pos:
            return 0
        start = self.pos
        with open(self.path, "rb") as f:
            p = self.pos
            while p + 12 <= target:
                f.seek(p)
                head = f.read(12)
                if len(head) < 12:
                    break
                btype = struct.unpack_from("<I", head, 0)[0]
                if btype == _BT_SHB:
                    bom = struct.unpack_from("<I", head, 8)[0]
                    self.endian = "<" if bom == _BOM else ">"
                btype, blen = struct.unpack_from(self.endian + "II", head, 0)
                if blen < 12 or blen % 4:
                    raise ValueError(f"{self.path}: corrupt block at byte {p}")
                if p + blen > size:
                    break
                if btype in (_BT_SHB, _BT_IDB):
                    f.seek(p)
                    block = f.read(blen)
                    if btype == _BT_SHB:
                        self.ts_units, self.linktypes = [], []
                    else:
                        self.linktypes.append(struct.unpack_from(self.endian + "H", block, 8)[0])
                        self.ts_units.append(_ts_unit_ns(_tsresol(block, 0, blen, self.endian)))
                elif btype in (_BT_EPB, _BT_SPB):
                    self.skipped_packets += 1
                p += blen
        self.pos = p
        self.skipped_bytes += p - start
        return p - start

    def read(self):
        """(u8, off, caplen, t_ns) of the complete packets appended since last call."""
        empty = (np.zeros(0, np.uint8), np.zeros(0, np.int64),
                 np.zeros(0, np.int64), np.zeros(0))
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return empty                # not created yet
        if size - self.pos < 12:
            return empty
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read(min(size - self.pos, self.max_read))

        off, caplen, t_ns = [], [], []
        p, n = 0, len(data)
        while p + 12 <= n:
            btype = struct.unpack_from("<I", data, p)[0]
            if btype == _BT_SHB:
                bom = struct.unpack_from("<I", data, p + 8)[0]
                self.endian = "<" if bom == _BOM else ">"
            e = self.endian
            btype, blen = struct.unpack_from(e + "II", data, p)
            if blen < 12 or blen % 4:
                raise ValueError(f"{self.path}: corrupt block at byte {self.pos + p}")
            if p + blen > n:
                break                   # dumpcap is still writing this block
            if btype == _BT_SHB:
                self.ts_units, self.linktypes = [], []
            elif btype == _BT_IDB:
                self.linktypes.append(struct.unpack_from(e + "H", data, p + 8)[0])
                self.ts_units.append(_ts_unit_ns(_tsresol(data, p, blen, e)))
            elif btype == _BT_EPB:
                iface, hi, lo, cl = struct.unpack_from(e + "IIII", data, p + 8)
                unit = self.ts_units[iface] if iface < len(self.ts_units) else 1e3
                off.append(p + 28)
                caplen.append(cl)
                t_ns.append(((hi << 32) | lo) * unit)
            elif btype == _BT_SPB:
                (orig,) = struct.unpack_from(e + "I", data, p + 8)
                off.append(p + 12)
                caplen.append(min(orig, blen - 16))
                t_ns.append(np.nan)
            p += blen
        self.pos += p
        self.n_packets += len(off)
        check_linktypes(self.linktypes, self.path)
        return (np.frombuffer(data, dtype=np.uint8), np.asarray(off, np.int64),
                np.asarray(caplen, np.int64), np.asarray(t_ns, np.float64))


def _tsresol(data, pos, blen, e):
    o, end = pos + 16, pos + blen - 4
    while o + 4 <= end:
        code, ln = struct.unpack_from(e + "HH", data, o)
        if code == 0:
            break
        if code == 9 and ln >= 1:
            return data[o + 4]
        o += 4 + ln + (-ln % 4)
    return 6


class StreamDecoder:
    """Hits of a growing capture, poll by poll, with state carried over."""

    def __init__(self, path, data_format="SRS", src_ips=None,
                 legacy_word_count=False, max_words=MAX_WORDS):
        self.follower = PcapngFollower(path)
        self.words = WordDecoder(data_format, legacy_word_count, max_words)
        self.legacy_word_count = legacy_word_count
        self.want = (np.array([ip_u32(s) for s in sorted(src_ips)], np.uint32)
                     if src_ips else None)
        self.fec_ips = set(src_ips or ())
        self.t_first_ns = None
        self.t_last_ns = None
        self.udp_payload_bytes = 0
        self.per_fec = {}
        self.frame_counters = set()

    @property
    def path(self):
        return self.follower.path

    def skip_to_tail(self, keep_bytes):
        """Drop the backlog beyond keep_bytes (see PcapngFollower.skip_to_tail).

        For monitoring under a flood: rates stay right over what is read, but
        hits are missing, and the first hits after the jump may carry the
        marker values from before it until their VMM's next marker.
        """
        return self.follower.skip_to_tail(keep_bytes)

    def poll(self):
        """(cols, info) for what arrived since the last poll; cols is None if no hit.

        cols: raw hit columns (RAW_COLUMNS) concatenated over the poll.
        info: packets, VM3 packets, payload bytes, first/last packet time (ns).
        """
        u8, off, caplen, t_ns = self.follower.read()
        info = {"packets": int(len(off)), "vm3_packets": 0, "udp_payload_bytes": 0,
                "t_first_ns": None, "t_last_ns": None}
        if not len(off):
            return None, info
        pstart, plen, src, keep = udp_payloads(u8, off, caplen)
        pstart, plen, src, t_ns = select_vm3(u8, pstart, plen, src, t_ns[keep],
                                             self.want, self.legacy_word_count)
        if not len(pstart):
            return None, info
        self.fec_ips.update(ip_str(int(v)) for v in np.unique(src))
        frame_counter, fec_id = packet_header(u8, pstart)
        self.frame_counters.update(np.unique(frame_counter).tolist()[:2])

        ok = np.isfinite(t_ns)
        if ok.any():
            t0, t1 = float(t_ns[ok].min()), float(t_ns[ok].max())
            info["t_first_ns"], info["t_last_ns"] = t0, t1
            self.t_first_ns = t0 if self.t_first_ns is None else min(self.t_first_ns, t0)
            self.t_last_ns = t1 if self.t_last_ns is None else max(self.t_last_ns, t1)
        info["vm3_packets"] = int(len(pstart))
        info["udp_payload_bytes"] = int(plen.sum())
        self.udp_payload_bytes += int(plen.sum())
        for f in np.unique(fec_id):
            sel = fec_id == f
            d = self.per_fec.setdefault(str(int(f)), {"packets": 0, "udp_payload_bytes": 0})
            d["packets"] += int(sel.sum())
            d["udp_payload_bytes"] += int(plen[sel].sum())

        parts = list(self.words.decode(u8, pstart, plen, frame_counter, fec_id))
        if not parts:
            return None, info
        cols = {k: (np.concatenate([p[k] for p in parts]) if len(parts) > 1
                    else parts[0][k]) for k in parts[0]}
        return cols, info

    def meta(self):
        """Cumulative meta, in the form decode() returns it."""
        m = self.words.meta(self.follower.n_packets, self.fec_ips)
        live = ((self.t_last_ns - self.t_first_ns) / 1e9
                if self.t_first_ns is not None else None)
        per_fec = {f: {**d, "hits": int(self.words.hits_per_fec[int(f)]),
                       "markers": int(self.words.markers_per_fec[int(f)])}
                   for f, d in self.per_fec.items()}
        m["capture"] = {"t_first_ns": self.t_first_ns, "t_last_ns": self.t_last_ns,
                        "live_time_s": live,
                        "udp_payload_bytes": self.udp_payload_bytes,
                        "frame_counter_changes": len(self.frame_counters) > 1,
                        "per_fec": per_fec}
        if self.legacy_word_count:
            m["legacy_word_count"] = True
        if self.follower.skipped_bytes:
            m["skipped"] = {"bytes": self.follower.skipped_bytes,
                            "packets": self.follower.skipped_packets}
        return m
