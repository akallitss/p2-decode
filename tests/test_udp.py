"""Decoding from a UDP socket: same hits as from a capture, drops counted."""

import socket
import time

import numpy as np
import pytest

import p2_decode
from p2_decode.testing import hit_word, marker_word, srs_payload, write_pcapng
from p2_decode.udp import UdpStreamDecoder, parse_address

pytestmark = pytest.mark.skipif(not hasattr(socket.socket, "recvmsg"),
                                reason="needs recvmsg (Linux)")


def _payloads(n=30):
    out = []
    for i in range(n):
        words = [marker_word(i % 3, 500 + i), hit_word(i % 3, i % 64, i, i, i)]
        words += [hit_word(5, 7, 1, 2, 3)] * (i % 4)
        out.append(srs_payload(i, 1 + i % 2, words))
    return out


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _send(port, payloads, pause=0.0):
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for p in payloads:
        tx.sendto(p, ("127.0.0.1", port))
        if pause:
            time.sleep(pause)
    tx.close()


def test_parse_address():
    assert parse_address("192.168.1.16:6006") == ("192.168.1.16", 6006)
    with pytest.raises(ValueError):
        parse_address("6006")


def test_udp_hits_equal_capture_hits(tmp_path):
    payloads = _payloads()
    port = _free_port()
    ud = UdpStreamDecoder(f"127.0.0.1:{port}")
    parts = []
    for chunk in (payloads[:7], payloads[7:19], payloads[19:]):    # several polls
        _send(port, chunk)
        time.sleep(0.05)
        cols, info = ud.poll()
        assert info["dropped"] == 0
        if cols is not None:
            parts.append(cols)
    got = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ref, _ = p2_decode.decode(write_pcapng(tmp_path / "c.pcapng", payloads))
    for c in p2_decode.RAW_COLUMNS:
        assert np.array_equal(got[c], ref[c].to_numpy()), c
    m = ud.meta()
    assert m["n_hits"] == len(ref) and m["udp"]["fraction_received"] == 1.0
    assert m["capture"]["live_time_s"] is not None and m["capture"]["live_time_s"] >= 0
    assert m["fec_ips"] == ["127.0.0.1"]
    ud.close()


def test_kernel_drops_are_counted():
    port = _free_port()
    ud = UdpStreamDecoder(f"127.0.0.1:{port}", rcvbuf=4096)      # tiny buffer
    sent = _payloads(300)
    _send(port, sent)                                            # nobody reads meanwhile
    time.sleep(0.05)
    got, dropped = 0, 0

    def drain():
        nonlocal got, dropped
        for _ in range(50):
            _, info = ud.poll()
            got += info["packets"]
            dropped += info["dropped"]
            if info["packets"] == 0:
                return
    drain()
    # the kernel attaches its drop count to the next datagram it can queue
    _send(port, sent[:1])
    time.sleep(0.05)
    drain()
    assert dropped > 0
    assert got + ud.receiver.dropped_total == len(sent) + 1
    assert 0 < ud.fraction_received < 1
    ud.close()


def test_read_budget_bounds_a_poll():
    port = _free_port()
    ud = UdpStreamDecoder(f"127.0.0.1:{port}")
    pl = [srs_payload(i, 1, [hit_word(1, 1, 1, 1, 1)] * 10) for i in range(50)]
    _send(port, pl)
    time.sleep(0.05)
    _, info = ud.poll(max_bytes=len(pl[0]) * 5)                  # equal sizes
    assert info["packets"] == 5
    ud.close()
