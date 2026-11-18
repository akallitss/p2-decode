"""get_packets / get_hits / PacketSource / HitSource: bounded batches, same hits."""

import socket
import threading
import time

import numpy as np
import pytest

import p2_decode
from p2_decode import HitSource, PacketSource, get_hits, get_packets
from p2_decode.testing import hit_word, marker_word, srs_payload, write_pcapng


def _payloads(n=40):
    out = []
    for i in range(n):
        words = [marker_word(i % 3, 100 + i), hit_word(i % 3, i % 64, i, i, i)]
        words += [hit_word((i + 1) % 3, 7, 1, 2, 3)] * (i % 4)   # hits before own marker
        out.append(srs_payload(i, 1 + i % 2, words))
    return out


@pytest.fixture
def payloads():
    return _payloads()


@pytest.fixture
def capture(tmp_path, payloads):
    # tsresol 3: packet i is stamped i ms
    return write_pcapng(tmp_path / "c.pcapng", payloads, tsresol=3)


def _same_raw(got, ref):
    assert len(got) == len(ref)
    for c in p2_decode.RAW_COLUMNS:
        assert np.array_equal(got[c].to_numpy(), ref[c].to_numpy()), c


def test_get_packets_npackets(capture, payloads):
    p = get_packets(file=capture, npackets=3)
    assert len(p) == 3
    assert [p.payload(i) for i in range(3)] == payloads[:3]
    assert p.fec_ips == ["10.0.0.2"]


def test_get_packets_whole_file(capture, payloads):
    p = get_packets(file=capture)
    assert [p.payload(i) for i in range(len(p))] == payloads


def test_consecutive_batches_cover_the_file_once(capture, payloads):
    src = PacketSource(file=capture)
    got = []
    for n in (1, 2, 5, 13, 100):
        got += [b.payload(i) for b in [src.get(npackets=n)] for i in range(len(b))]
    assert got == payloads
    assert len(src.get(npackets=5)) == 0 and src.n_packets == len(payloads)


def test_duration_is_data_time(capture):
    src = PacketSource(file=capture)
    a = src.get(duration=0.005)                 # packets at 0..4 ms
    b = src.get(duration=0.005)                 # 5..9 ms
    assert len(a) == 5 and len(b) == 5
    assert (b.t_ns - a.t_ns[0]).min() == pytest.approx(5e6)


def test_first_bound_wins(capture):
    assert len(get_packets(file=capture, npackets=3, duration=0.010)) == 3
    assert len(get_packets(file=capture, npackets=30, duration=0.010)) == 10


def test_decode_packets_equals_decode_file(capture):
    ref, meta = p2_decode.decode(capture)
    hits, m = p2_decode.decode(get_packets(file=capture))
    _same_raw(hits, ref)
    assert list(hits.columns) == p2_decode.COLUMNS
    assert m["n_hits"] == meta["n_hits"]


@pytest.mark.parametrize("sizes", [(1,) * 40, (3, 7, 11, 19), (40,)])
def test_hit_source_batches_equal_one_decode(capture, sizes):
    ref, meta = p2_decode.decode(capture)
    src = HitSource(file=capture)
    parts = [src.get(npackets=n) for n in sizes]
    import pandas as pd
    _same_raw(pd.concat(parts, ignore_index=True), ref)
    assert src.meta()["n_hits"] == meta["n_hits"]
    assert src.meta()["capture"]["live_time_s"] == pytest.approx(0.039)


def test_get_hits(capture):
    ref, _ = p2_decode.decode(capture, max_packets=10)
    _same_raw(get_hits(file=capture, npackets=10), ref)
    _same_raw(get_hits(file=capture, duration=0.010), ref)


def test_src_ips_filter(capture):
    assert len(get_packets(file=capture, src_ips=["10.0.0.9"])) == 0
    assert len(get_hits(file=capture, src_ips=["10.0.0.9"])) == 0


def test_bad_arguments(capture):
    with pytest.raises(ValueError):
        PacketSource()
    with pytest.raises(ValueError):
        PacketSource(file=capture, port=6006)
    with pytest.raises(ValueError):
        get_packets(file=capture, npackets=-1)
    assert len(get_packets(file=capture, npackets=0)) == 0


def test_growing_file_is_followed(tmp_path, capture):
    data = open(capture, "rb").read()
    grow = tmp_path / "grow.pcapng"
    grow.write_bytes(data[: len(data) * 6 // 10])       # cut inside a block
    src = HitSource(file=grow, follow=True)
    a = src.get(timeout=0.3)
    with open(grow, "ab") as f:
        f.write(data[len(data) * 6 // 10:])
    b = src.get(timeout=0.3)
    import pandas as pd
    ref, _ = p2_decode.decode(capture)
    assert 0 < len(a) < len(ref)
    _same_raw(pd.concat([a, b], ignore_index=True), ref)


# --- from a UDP port -----------------------------------------------------------

udp = pytest.mark.skipif(not hasattr(socket.socket, "recvmsg"), reason="needs recvmsg (Linux)")


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _send_later(port, payloads, delay=0.1, pause=0.0):
    def run():
        time.sleep(delay)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for p in payloads:
            tx.sendto(p, ("127.0.0.1", port))
            if pause:
                time.sleep(pause)
        tx.close()
    t = threading.Thread(target=run)
    t.start()
    return t


@udp
def test_port_npackets_equals_file(capture, payloads):
    port = _free_port()
    with HitSource(port=f"127.0.0.1:{port}") as src:
        t = _send_later(port, payloads)
        a = src.get(npackets=15, timeout=3)
        b = src.get(npackets=25, timeout=3)
        t.join()
        import pandas as pd
        ref, _ = p2_decode.decode(capture)
        _same_raw(pd.concat([a, b], ignore_index=True), ref)
        assert src.meta()["udp"]["dropped"] == 0
        assert src.meta()["fec_ips"] == ["127.0.0.1"]


@udp
def test_port_duration_returns_on_time(payloads):
    port = _free_port()
    t = _send_later(port, payloads[:5], delay=0.05)
    t0 = time.monotonic()
    p = get_packets(port=f"127.0.0.1:{port}", duration=0.5)
    took = time.monotonic() - t0
    t.join()
    assert len(p) == 5
    assert took < 2.0          # gives up at duration + 1 s when the stream stops


# --- command line ----------------------------------------------------------------

def test_cli_file_paths(capture, tmp_path, capsys):
    from p2_decode.cli import main
    assert main([str(capture)]) == 0                                   # unchanged path
    assert main(["--file", str(capture), "--npackets", "10"]) == 0
    out = capsys.readouterr().out
    assert "n_hits" in out
    store = tmp_path / "st"
    assert main(["--file", str(capture), "--duration", "0.010",
                 "--out-dir", str(store)]) == 0
    hits, meta = p2_decode.load_frame(str(store))
    ref, _ = p2_decode.decode(capture, max_packets=10)
    _same_raw(hits, ref)
    assert meta["schema_version"] == p2_decode.SCHEMA_VERSION


def test_cli_refuses(capture):
    from p2_decode.cli import main
    for argv in ([], ["--port", "6006"], [str(capture), "--port", "6006"]):
        with pytest.raises(SystemExit):
            main(argv)


@udp
def test_cli_port(capture, payloads, capsys):
    from p2_decode.cli import main
    port = _free_port()
    t = _send_later(port, payloads, delay=0.3)
    assert main(["--port", f"127.0.0.1:{port}", "--npackets", "40"]) == 0
    t.join()
    ref, meta = p2_decode.decode(capture)
    assert f"n_hits             {meta['n_hits']}" in capsys.readouterr().out
