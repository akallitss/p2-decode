"""Decoding straight from the network: the FEC's UDP stream, no file in between.

    ud = UdpStreamDecoder("192.168.1.16:6006")      # the data NIC and port
    cols, info = ud.poll(max_bytes=16 << 20)        # hits received since last poll

A plain UDP socket on the port the FEC sends to. The kernel hands every
datagram both to libpcap (dumpcap keeps recording everything) and to this
socket, so reading here never takes data away from the recording. If this
reader is slower than the stream, the kernel drops datagrams for THIS socket
only, and counts them (SO_RXQ_OVFL): `info["dropped"]` and the cumulative
`fraction_received` say how much of the stream was decoded, so rates can be
scaled to the full stream. Packet times are the kernel's receive timestamps
(SO_TIMESTAMPNS), so live time is measured exactly as for a capture file.

Linux only. Needs no privileges; the port must not be bound by anything else.
"""

import socket
import struct
import time

import numpy as np

from .pcapng import ip_str, ip_u32
from .srs import MAX_WORDS, WordDecoder, packet_header, select_vm3

# Linux socket options (not all are exposed by Python's socket module)
SO_TIMESTAMPNS = getattr(socket, "SO_TIMESTAMPNS", 35)
SO_RXQ_OVFL = getattr(socket, "SO_RXQ_OVFL", 40)
RCVBUF = 4 * 1024 * 1024        # what an unprivileged user gets (net.core.rmem_max)
_CMSG_SPACE = socket.CMSG_SPACE(16) + socket.CMSG_SPACE(4)


def parse_address(addr):
    """'192.168.1.16:6006' -> ('192.168.1.16', 6006)."""
    host, _, port = str(addr).rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"expected HOST:PORT, got {addr!r}")
    return host, int(port)


class UdpReceiver:
    """Non-blocking UDP socket with kernel timestamps and a drop counter."""

    def __init__(self, address, rcvbuf=RCVBUF):
        self.host, self.port = parse_address(address)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        s.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        s.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
        s.bind((self.host, self.port))
        s.setblocking(False)
        self.sock = s
        self.rcvbuf = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.dropped_total = 0           # kernel's cumulative count for this socket

    def read(self, max_bytes, max_seconds=1.0):
        """Datagrams waiting now: (payloads, t_ns, src_ip_u32), at most max_bytes.

        Stops when the socket is empty, max_bytes are read, or max_seconds have
        passed, so a flood cannot keep the caller reading forever; whatever is
        not read is dropped and counted by the kernel.
        """
        payloads, times, srcs = [], [], []
        n, t_end = 0, time.monotonic() + max_seconds
        while n < max_bytes and time.monotonic() < t_end:
            try:
                data, anc, _flags, addr = self.sock.recvmsg(65536, _CMSG_SPACE)
            except BlockingIOError:
                break
            t_ns = np.nan
            for level, kind, cdata in anc:
                if level != socket.SOL_SOCKET:
                    continue
                if kind == SO_TIMESTAMPNS and len(cdata) >= 16:
                    sec, nsec = struct.unpack("qq", cdata[:16])
                    t_ns = float(sec) * 1e9 + nsec
                elif kind == SO_RXQ_OVFL and len(cdata) >= 4:
                    self.dropped_total = struct.unpack("I", cdata[:4])[0]
            payloads.append(data)
            times.append(t_ns)
            srcs.append(int(ip_u32(addr[0])))
            n += len(data)
        return payloads, np.asarray(times, np.float64), np.asarray(srcs, np.uint32)

    def close(self):
        self.sock.close()


class UdpStreamDecoder:
    """Hits from the FEC's UDP stream, poll by poll, with state carried over.

    The same WordDecoder as the file paths, so a datagram decodes to the same
    hits whether it is read here or from dumpcap's capture of it.
    """

    def __init__(self, address, data_format="SRS", src_ips=None,
                 legacy_word_count=False, max_words=MAX_WORDS, rcvbuf=RCVBUF):
        self.receiver = UdpReceiver(address, rcvbuf)
        self.words = WordDecoder(data_format, legacy_word_count, max_words)
        self.legacy_word_count = legacy_word_count
        self.want = (np.array([ip_u32(s) for s in sorted(src_ips)], np.uint32)
                     if src_ips else None)
        self.fec_ips = set(src_ips or ())
        self.received = 0
        self.received_bytes = 0
        self._dropped_seen = 0
        self.t_first_ns = None
        self.t_last_ns = None
        self.per_fec = {}

    @property
    def address(self):
        return f"{self.receiver.host}:{self.receiver.port}"

    def poll(self, max_bytes=16 << 20, max_seconds=1.0):
        """(cols, info) for the datagrams read now; cols is None if no hit.

        info: packets and bytes received, `dropped` by the kernel since the
        previous poll, first/last receive time (ns since the epoch).
        """
        payloads, t_ns, src = self.receiver.read(max_bytes, max_seconds)
        dropped = self.receiver.dropped_total - self._dropped_seen
        self._dropped_seen = self.receiver.dropped_total
        info = {"packets": len(payloads), "dropped": int(dropped),
                "udp_payload_bytes": 0, "vm3_packets": 0,
                "t_first_ns": None, "t_last_ns": None}
        self.received += len(payloads)
        if not payloads:
            return None, info
        plen = np.fromiter((len(p) for p in payloads), np.int64, len(payloads))
        self.received_bytes += int(plen.sum())
        pstart = np.concatenate([[0], np.cumsum(plen)[:-1]]).astype(np.int64)
        u8 = np.frombuffer(b"".join(payloads), dtype=np.uint8)
        pstart, plen, src, t_ns = select_vm3(u8, pstart, plen, src, t_ns, self.want,
                                             self.legacy_word_count)
        if not len(pstart):
            return None, info
        self.fec_ips.update(ip_str(int(v)) for v in np.unique(src))
        frame_counter, fec_id = packet_header(u8, pstart)
        ok = np.isfinite(t_ns)
        if ok.any():
            t0, t1 = float(t_ns[ok].min()), float(t_ns[ok].max())
            info["t_first_ns"], info["t_last_ns"] = t0, t1
            self.t_first_ns = t0 if self.t_first_ns is None else min(self.t_first_ns, t0)
            self.t_last_ns = t1 if self.t_last_ns is None else max(self.t_last_ns, t1)
        info["vm3_packets"] = int(len(pstart))
        info["udp_payload_bytes"] = int(plen.sum())
        for f in np.unique(fec_id):
            sel = fec_id == f
            d = self.per_fec.setdefault(str(int(f)), {"packets": 0, "udp_payload_bytes": 0})
            d["packets"] += int(sel.sum())
            d["udp_payload_bytes"] += int(plen[sel].sum())
        parts = list(self.words.decode(u8, pstart, plen, frame_counter, fec_id,
                                       pkt_time_ns=t_ns))
        if not parts:
            return None, info
        cols = {k: (np.concatenate([p[k] for p in parts]) if len(parts) > 1
                    else parts[0][k]) for k in parts[0]}
        return cols, info

    @property
    def fraction_received(self):
        """Datagrams decoded / datagrams that reached the socket, so far."""
        total = self.received + self.receiver.dropped_total
        return self.received / total if total else None

    def meta(self):
        m = self.words.meta(self.received, self.fec_ips)
        live = ((self.t_last_ns - self.t_first_ns) / 1e9
                if self.t_first_ns is not None else None)
        m["capture"] = {"t_first_ns": self.t_first_ns, "t_last_ns": self.t_last_ns,
                        "live_time_s": live, "udp_payload_bytes": self.received_bytes,
                        "per_fec": self.per_fec}
        m["udp"] = {"address": self.address, "received": self.received,
                    "dropped": self.receiver.dropped_total,
                    "fraction_received": self.fraction_received,
                    "rcvbuf": self.receiver.rcvbuf}
        return m

    def close(self):
        self.receiver.close()
