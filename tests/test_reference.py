"""A real capture (tests/data/lab_run263_head.pcapng) against its reference hits.

Every way into p2-decode -- whole file, column store, bounded batches, and the
same packets replayed through a UDP port -- must give the reference exactly.
See tests/data/make_reference.py for where the capture and reference come from.
"""

import json
import os
import socket
import threading
import time

import numpy as np
import pandas as pd
import pytest

import p2_decode
from p2_decode import HitSource, get_packets

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CAPTURE = os.path.join(DATA, "lab_run263_head.pcapng")


@pytest.fixture(scope="module")
def ref():
    with np.load(os.path.join(DATA, "lab_run263_head_hits.npz")) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def ref_meta():
    with open(os.path.join(DATA, "lab_run263_head_meta.json")) as f:
        return json.load(f)


def _check(got, ref):
    got = got if isinstance(got, dict) else {c: got[c].to_numpy() for c in got.columns}
    for c in p2_decode.RAW_COLUMNS:
        assert got[c].dtype == ref[c].dtype, c
        assert np.array_equal(got[c], ref[c]), c


def test_decode(ref, ref_meta):
    hits, meta = p2_decode.decode(CAPTURE)
    _check(hits, ref)
    for k, v in ref_meta.items():
        got = meta["capture"]["live_time_s"] if k == "live_time_s" else meta[k]
        assert got == pytest.approx(v) if isinstance(v, float) else got == v, k


def test_column_store(ref, tmp_path):
    p2_decode.decode_to(CAPTURE, str(tmp_path / "st"))
    hits, _ = p2_decode.load_frame(str(tmp_path / "st"))
    _check(hits, ref)


@pytest.mark.parametrize("max_words", [1000, 7777, p2_decode.MAX_WORDS])
def test_chunk_size_does_not_matter(ref, max_words):
    hits, _ = p2_decode.decode(CAPTURE, max_words=max_words)
    _check(hits, ref)


def test_batches(ref):
    src = HitSource(file=CAPTURE)
    parts = [src.get(npackets=n) for n in (1, 4, 9, 100)]
    _check(pd.concat(parts, ignore_index=True), ref)
    src = HitSource(file=CAPTURE)
    parts = [src.get(duration=0.05) for _ in range(10)]
    _check(pd.concat(parts, ignore_index=True), ref)


@pytest.mark.skipif(not hasattr(socket.socket, "recvmsg"), reason="needs recvmsg (Linux)")
def test_replayed_through_a_port(ref):
    packets = get_packets(file=CAPTURE)
    payloads = [packets.payload(i) for i in range(len(packets))]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with HitSource(port=f"127.0.0.1:{port}") as src:
        def send():
            time.sleep(0.1)
            tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for p in payloads:
                tx.sendto(p, ("127.0.0.1", port))
                time.sleep(0.001)
            tx.close()
        t = threading.Thread(target=send)
        t.start()
        hits = src.get(npackets=len(payloads), timeout=5)
        t.join()
        assert src.meta()["udp"]["dropped"] == 0
    _check(hits, ref)
