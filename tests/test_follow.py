"""Streaming decode of a growing capture == decode of the finished file.

The capture is copied into a new file in random byte pieces (cutting through
blocks, headers and packets), with a poll after each piece. Real captures are
used too when P2_DECODE_CAPTURES is set (see test_legacy.py).
"""

import os

import numpy as np
import pytest

import p2_decode
from p2_decode.testing import hit_word, marker_word, srs_payload, write_pcapng

CAPTURES = [p for p in os.environ.get("P2_DECODE_CAPTURES", "").split(":") if p]


def _stream(src, dst, n_pieces, seed=0, data_format="SRS", **kw):
    """Grow `dst` from `src` in random pieces, polling after each. Returns (cols, meta)."""
    raw = open(src, "rb").read()
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(1, len(raw)), size=min(n_pieces, len(raw) - 1),
                              replace=False))
    sd = p2_decode.StreamDecoder(dst, data_format=data_format, **kw)
    assert sd.poll()[0] is None                         # file does not exist yet
    parts, prev = [], 0
    with open(dst, "wb") as f:
        for c in list(cuts) + [len(raw)]:
            f.write(raw[prev:c])
            f.flush()
            prev = c
            cols, _ = sd.poll()
            if cols is not None:
                parts.append(cols)
    assert sd.poll()[0] is None                         # nothing left over
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return cols, sd.meta()


def _same(path, cols, meta, data_format="SRS", **kw):
    ref, ref_meta = p2_decode.decode(path, data_format=data_format, **kw)
    for c in p2_decode.RAW_COLUMNS:
        a, b = ref[c].to_numpy(), cols[c]
        assert a.dtype == b.dtype and np.array_equal(a, b), c
    for k in ("n_packets", "n_vm3_packets", "n_hits", "n_words", "n_markers",
              "n_invalid_offset", "offset_constant", "fec_ips", "capture"):
        assert meta[k] == ref_meta[k], k


@pytest.fixture
def capture(tmp_path):
    payloads = []
    for i in range(40):
        words = [marker_word(i % 4, 1000 + i), hit_word(i % 4, i % 64, i, i, i)]
        words += [hit_word(4, 1, 2, 3, 4)] * (i % 5)
        payloads.append(srs_payload(i, 1 + i % 2, words))
    return write_pcapng(tmp_path / "c.pcapng", payloads, tsresol=9)


@pytest.mark.parametrize("n_pieces", [1, 7, 300, 5000])
def test_synthetic_stream_equals_whole_file(capture, tmp_path, n_pieces):
    cols, meta = _stream(capture, tmp_path / "grow.pcapng", n_pieces)
    _same(capture, cols, meta)


def test_legacy_word_count_stream(capture, tmp_path):
    cols, meta = _stream(capture, tmp_path / "g.pcapng", 50, legacy_word_count=True)
    _same(capture, cols, meta, legacy_word_count=True)


def test_half_written_block_waits(capture, tmp_path):
    raw = open(capture, "rb").read()
    dst = tmp_path / "h.pcapng"
    sd = p2_decode.StreamDecoder(dst)
    dst.write_bytes(raw[:len(raw) - 10])                # last block cut short
    sd.poll()
    pos = sd.follower.pos
    assert pos < len(raw) - 10
    with open(dst, "ab") as f:
        f.write(raw[len(raw) - 10:])
    sd.poll()
    assert sd.follower.pos == len(raw)


@pytest.mark.skipif(not CAPTURES, reason="set P2_DECODE_CAPTURES")
@pytest.mark.parametrize("data_format", ["SRS", "TRG"])
@pytest.mark.parametrize("path", CAPTURES, ids=os.path.basename)
def test_real_capture_stream(path, tmp_path, data_format):
    cols, meta = _stream(path, tmp_path / "grow.pcapng", 40, seed=1,
                         data_format=data_format)
    _same(path, cols, meta, data_format=data_format)


def test_skip_to_tail(capture, tmp_path):
    raw = open(capture, "rb").read()
    dst = tmp_path / "s.pcapng"
    dst.write_bytes(raw)
    sd = p2_decode.StreamDecoder(dst)
    assert sd.follower.backlog() == len(raw)
    skipped = sd.skip_to_tail(keep_bytes=300)
    assert skipped > 0 and sd.follower.backlog() <= 300 + 200      # lands on a block edge
    cols, _ = sd.poll()
    whole, _ = p2_decode.decode(capture)
    n = len(cols["vmm"])
    assert 0 < n < len(whole)
    # what is read after the jump is the tail of the file, hit for hit
    for c in ("vmm", "ch", "adc", "bcid", "tdc", "fec"):
        assert np.array_equal(cols[c], whole[c].to_numpy()[-n:]), c
    m = sd.meta()
    assert m["skipped"]["packets"] + m["n_packets"] == 40
    assert m["capture"]["t_first_ns"] is not None                 # IDB units kept


def test_skip_nothing_when_caught_up(capture, tmp_path):
    dst = tmp_path / "s.pcapng"
    dst.write_bytes(open(capture, "rb").read())
    sd = p2_decode.StreamDecoder(dst)
    sd.poll()
    assert sd.skip_to_tail(keep_bytes=0) == 0
