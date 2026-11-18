"""Packets and hits from a file or a UDP port, a bounded batch at a time.

    from p2_decode import get_packets, get_hits, decode

    packets = get_packets(file="cap.pcapng", npackets=1000)   # the first 1000
    hits = decode(packets)[0]                                 # ... decoded
    hits = get_hits(file="cap.pcapng", duration=2.0)          # first 2 s of data
    hits = get_hits(port=6006, duration=5.0)                  # 5 s off the network

To carry on from where the previous call stopped, keep a source:

    src = HitSource(file="cap.pcapng")      # or port="192.168.1.16:6006"
    a = src.get(npackets=1000)              # packets 0..999, decoded
    b = src.get(duration=1.0)               # the next second of data
    src.meta()                              # cumulative counts, like decode()'s

Bounds: `npackets` counts VM3 packets (SRS data packets from the FEC, after
the `src_ips` filter); `duration` is in seconds of DATA time, measured on the
packet timestamps (capture time for files, kernel receive time for a port)
from the first packet of the call. With both, the first bound reached wins.
With neither, a file is read to its end, and a port for DEFAULT_PORT_DURATION_S.

A HitSource decodes with one WordDecoder for its whole life, so hits are
identical to one decode() of the same packets, whatever the batch sizes.
decode(packets) on a bare Packets batch starts with no marker seen: hits
before their VMM's first marker in the batch get srs_timestamp 0.

A file is read in blocks of at most `read_bytes`; with `follow=True` it may
still be growing (dumpcap writing it), and get() waits up to `timeout` seconds
for more data before returning what it has.
"""

import time

import numpy as np

from .follow import PcapngFollower
from .pcapng import ip_str, ip_u32, udp_payloads
from .srs import MAX_WORDS, WordDecoder, packet_header, select_vm3
from .udp import RCVBUF, UdpReceiver

DEFAULT_PORT_DURATION_S = 1.0
READ_BYTES = 32 * 1024 * 1024


class Packets:
    """A batch of VM3 payloads: one contiguous buffer plus per-packet arrays.

    u8      uint8 buffer holding the payloads back to back
    pstart  int64 start of each payload in u8
    plen    int64 length of each payload (bytes)
    src     uint32 source IPv4 of each packet
    t_ns    float64 packet time, ns since the epoch (NaN if unknown)
    """

    def __init__(self, u8, pstart, plen, src, t_ns):
        self.u8 = u8
        self.pstart = np.asarray(pstart, np.int64)
        self.plen = np.asarray(plen, np.int64)
        self.src = np.asarray(src, np.uint32)
        self.t_ns = np.asarray(t_ns, np.float64)

    def __len__(self):
        return len(self.pstart)

    def __repr__(self):
        return f"<Packets: {len(self)} VM3 packets, {int(self.plen.sum())} payload bytes>"

    def payload(self, i):
        """The i-th UDP payload as bytes (16-byte SRS header + 6-byte words)."""
        s = int(self.pstart[i])
        return self.u8[s:s + int(self.plen[i])].tobytes()

    @property
    def fec_ips(self):
        return sorted(ip_str(int(v)) for v in np.unique(self.src))

    @property
    def header(self):
        """(frame_counter, fec_id) of each packet."""
        return packet_header(self.u8, self.pstart)

    def slice(self, lo, hi):
        return Packets(self.u8, self.pstart[lo:hi], self.plen[lo:hi],
                       self.src[lo:hi], self.t_ns[lo:hi])

    @classmethod
    def empty(cls):
        z = np.zeros(0, np.int64)
        return cls(np.zeros(0, np.uint8), z, z, np.zeros(0, np.uint32), np.zeros(0))

    @classmethod
    def compact(cls, u8, pstart, plen, src, t_ns):
        """Copy the payloads out of a larger buffer (a pcapng read) into their own."""
        if len(pstart) == 0:
            return cls.empty()
        buf = np.concatenate([u8[s:s + n] for s, n in zip(pstart.tolist(), plen.tolist())])
        new = np.concatenate([[0], np.cumsum(plen)[:-1]]).astype(np.int64)
        return cls(buf, new, plen, src, t_ns)

    @classmethod
    def concat(cls, parts):
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        if len(parts) == 1:
            return parts[0]
        bufs, starts, base = [], [], 0
        for p in parts:
            lo = int(p.pstart[0])
            hi = int(p.pstart[-1] + p.plen[-1])
            bufs.append(p.u8[lo:hi])
            starts.append(p.pstart - lo + base)
            base += hi - lo
        return cls(np.concatenate(bufs), np.concatenate(starts),
                   np.concatenate([p.plen for p in parts]),
                   np.concatenate([p.src for p in parts]),
                   np.concatenate([p.t_ns for p in parts]))


def _address(port):
    """6006 or "6006" -> all interfaces; "HOST:PORT" as given."""
    s = str(port)
    return f"0.0.0.0:{s}" if s.isdigit() else s


class PacketSource:
    """VM3 packets from a capture file or a UDP port, in order, call after call."""

    def __init__(self, file=None, port=None, src_ips=None, follow=False,
                 legacy_word_count=False, read_bytes=READ_BYTES, rcvbuf=RCVBUF):
        if (file is None) == (port is None):
            raise ValueError("give exactly one of file= or port=")
        self.file, self.port = file, port
        self.follow = follow
        self.legacy_word_count = legacy_word_count
        self.want = (np.array([ip_u32(s) for s in sorted(src_ips)], np.uint32)
                     if src_ips else None)
        self.follower = PcapngFollower(file, max_read=read_bytes) if file is not None else None
        self.receiver = UdpReceiver(_address(port), rcvbuf) if port is not None else None
        self._pending = []          # Packets read but not handed out yet
        self.n_packets = 0          # VM3 packets handed out so far
        self.at_end = False         # a file (not followed) is fully read

    # -- raw reads -----------------------------------------------------------
    def _read(self, max_seconds):
        """One read from the file or the socket, VM3-filtered and compacted."""
        if self.follower is not None:
            u8, off, caplen, t_ns = self.follower.read()
            if not len(off):
                return None
            pstart, plen, src, keep = udp_payloads(u8, off, caplen)
            t_ns = t_ns[keep]
        else:
            payloads, t_ns, src = self.receiver.read(16 << 20, max_seconds)
            if not payloads:
                return None
            plen = np.fromiter((len(p) for p in payloads), np.int64, len(payloads))
            pstart = np.concatenate([[0], np.cumsum(plen)[:-1]]).astype(np.int64)
            u8 = np.frombuffer(b"".join(payloads), dtype=np.uint8)
            t_ns, src = np.asarray(t_ns, np.float64), np.asarray(src, np.uint32)
        pstart, plen, src, t_ns = select_vm3(u8, pstart, plen, src, t_ns,
                                             self.want, self.legacy_word_count)
        if self.follower is not None:
            return Packets.compact(u8, pstart, plen, src, t_ns)
        return Packets(u8, pstart, plen, src, t_ns)

    # -- bounded batches -----------------------------------------------------
    def _waits(self):
        """True if running out of data means waiting, not the end (port, followed file)."""
        return self.receiver is not None or self.follow

    def get(self, npackets=None, duration=None, timeout=None):
        """The next VM3 packets, up to `npackets` and/or `duration` s of data time.

        Packets past the bound are kept for the next call. A file not followed
        is read until a bound is met or the file ends, however long that takes.
        A port, or a followed file, returns what it has once `timeout` seconds
        of wall-clock time have passed (default: duration + 1 s for a port, or
        10 s with only npackets; 2 s for a followed file).
        """
        if npackets is not None and npackets < 0:
            raise ValueError("npackets must be >= 0")
        if npackets is None and duration is None and self.receiver is not None:
            duration = DEFAULT_PORT_DURATION_S
        if timeout is None:
            if self.receiver is not None:
                timeout = duration + 1.0 if duration is not None else 10.0
            else:
                timeout = 2.0
        t_wall_end = time.monotonic() + timeout

        parts, state = [], {"n": 0, "t0": None}
        queue, self._pending = list(self._pending), []
        if npackets == 0:
            self._pending = queue
            return Packets.empty()
        while True:
            if queue:
                if self._take(queue.pop(0), parts, state, npackets, duration):
                    self._pending += queue
                    break
                continue
            if self.at_end:
                break
            if self._waits() and time.monotonic() >= t_wall_end:
                break
            got = self._read(max_seconds=min(0.2, max(0.0, t_wall_end - time.monotonic())))
            if got is None:
                if not self._waits():
                    self.at_end = True          # a complete file: nothing more to come
                elif self.follower is not None:
                    time.sleep(0.05)
                continue
            if len(got):
                queue.append(got)
        out = Packets.concat(parts)
        self.n_packets += len(out)
        return out

    def _take(self, p, parts, state, npackets, duration):
        """Add packets `p` to the batch; True (rest kept for later) if a bound is met."""
        cut = None
        if npackets is not None and len(p) >= npackets - state["n"]:
            cut = npackets - state["n"]
        if duration is not None and len(p):
            ok = np.isfinite(p.t_ns)
            if state["t0"] is None and ok.any():
                state["t0"] = float(p.t_ns[ok][0])
            if state["t0"] is not None:
                over = np.flatnonzero(ok & (p.t_ns - state["t0"] >= duration * 1e9))
                if len(over):
                    cut = int(over[0]) if cut is None else min(cut, int(over[0]))
        if cut is None:
            parts.append(p)
            state["n"] += len(p)
            return False
        parts.append(p.slice(0, cut))
        if cut < len(p):
            self._pending.insert(0, p.slice(cut, len(p)))
        return True

    def close(self):
        if self.receiver is not None:
            self.receiver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class HitSource(PacketSource):
    """Decoded hits from a file or a UDP port, call after call, marker state kept."""

    def __init__(self, file=None, port=None, data_format="SRS", src_ips=None,
                 calibration=None, follow=False, legacy_word_count=False,
                 max_words=MAX_WORDS, **kw):
        super().__init__(file=file, port=port, src_ips=src_ips, follow=follow,
                         legacy_word_count=legacy_word_count, **kw)
        self.words = WordDecoder(data_format, legacy_word_count, max_words)
        self.calibration = calibration
        self.fec_ips = set(src_ips or ())
        self.t_first_ns = None
        self.t_last_ns = None

    def get_cols(self, npackets=None, duration=None, timeout=None):
        """(raw hit columns or None, the Packets they came from)."""
        packets = PacketSource.get(self, npackets, duration, timeout)
        if not len(packets):
            return None, packets
        self.fec_ips.update(packets.fec_ips)
        ok = np.isfinite(packets.t_ns)
        if ok.any():
            t0, t1 = float(packets.t_ns[ok].min()), float(packets.t_ns[ok].max())
            self.t_first_ns = t0 if self.t_first_ns is None else min(self.t_first_ns, t0)
            self.t_last_ns = t1 if self.t_last_ns is None else max(self.t_last_ns, t1)
        return _decode_cols(self.words, packets), packets

    def get(self, npackets=None, duration=None, timeout=None):
        """The hits of the next packets (bounds as PacketSource.get), as a DataFrame."""
        cols, _ = self.get_cols(npackets, duration, timeout)
        return _frame(cols, self.calibration)

    def meta(self):
        """Cumulative meta, in the form decode() returns it."""
        m = self.words.meta(self.n_packets, self.fec_ips)
        live = ((self.t_last_ns - self.t_first_ns) / 1e9
                if self.t_first_ns is not None else None)
        m["capture"] = {"t_first_ns": self.t_first_ns, "t_last_ns": self.t_last_ns,
                        "live_time_s": live}
        if self.receiver is not None:
            m["udp"] = {"dropped": int(self.receiver.dropped_total)}
        return m


def _decode_cols(words, packets):
    frame_counter, fec_id = packets.header
    parts = list(words.decode(packets.u8, packets.pstart, packets.plen,
                              frame_counter, fec_id))
    if not parts:
        return None
    return {k: (np.concatenate([p[k] for p in parts]) if len(parts) > 1
                else parts[0][k]) for k in parts[0]}


def _frame(cols, calibration=None):
    import pandas as pd

    from .calibration import derive
    from .schema import COLUMNS, empty_hits
    if cols is None:
        return empty_hits()
    derive(cols, calibration)
    return pd.DataFrame({c: cols[c] for c in COLUMNS})


def decode_packets(packets, data_format="SRS", calibration=None,
                   legacy_word_count=False, max_words=MAX_WORDS):
    """Hits of a Packets batch (fresh marker state). Returns (hits, meta)."""
    words = WordDecoder(data_format, legacy_word_count, max_words)
    hits = _frame(_decode_cols(words, packets), calibration)
    return hits, words.meta(len(packets), packets.fec_ips)


def get_packets(file=None, port=None, npackets=None, duration=None, src_ips=None,
                timeout=None, **kw):
    """One bounded batch of VM3 packets from the start of a file, or from a port.

    For consecutive batches keep a PacketSource instead.
    """
    with PacketSource(file=file, port=port, src_ips=src_ips, **kw) as s:
        return s.get(npackets, duration, timeout)


def get_hits(file=None, port=None, npackets=None, duration=None, data_format="SRS",
             src_ips=None, calibration=None, timeout=None, **kw):
    """The hits of one bounded batch of packets, as a DataFrame (see get_packets).

    For consecutive batches keep a HitSource instead; for a whole large file
    use decode_to, which streams to disk.
    """
    with HitSource(file=file, port=port, data_format=data_format, src_ips=src_ips,
                   calibration=calibration, **kw) as s:
        return s.get(npackets, duration, timeout)
