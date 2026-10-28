import numpy as np
import pytest

import p2_decode
from p2_decode.srs import gray2bin
from p2_decode.testing import hit_word, marker_word, srs_payload, write_pcapng


@pytest.fixture
def capture(tmp_path):
    words1 = [marker_word(2, 1000),
              hit_word(2, 10, 300, 1234, 77),
              hit_word(3, 63, 1023, 4095, 255, ot=1)]
    words2 = [hit_word(2, 11, 5, 0, 0, offset=1),
              marker_word(2, 2000),
              hit_word(2, 12, 6, 1, 1)]
    return write_pcapng(tmp_path / "c.pcapng",
                        [srs_payload(7, 1, words1), srs_payload(8, 1, words2)])


def test_hit_fields(capture):
    hits, meta = p2_decode.decode(capture)
    assert list(hits.columns) == p2_decode.COLUMNS
    assert meta["n_hits"] == 4 and meta["n_packets"] == 2
    assert meta["fec_ips"] == ["10.0.0.2"]
    h = hits.iloc[0]
    assert (h.fec, h.vmm, h.ch, h.adc, h.bcid, h.tdc) == (1, 2, 10, 300, 1234, 77)
    assert h.time == 7
    h = hits.iloc[1]
    assert (h.vmm, h.ch, h.adc, h.bcid, h.tdc, h.over_threshold) == \
        (3, 63, 1023, 4095, 255, True)
    assert hits.iloc[2].offset == 1 and hits.iloc[2].time == 8
    for c, dt in p2_decode.COLUMN_DTYPES.items():
        assert hits[c].dtype == np.dtype(dt), c


def test_markers_follow_own_vmm(capture):
    hits, _ = p2_decode.decode(capture)
    # VMM 2 sees its own markers in word order, across packets
    assert hits.srs_timestamp.tolist() == [1000, 0, 1000, 2000]


def test_derived_columns(capture):
    hits, _ = p2_decode.decode(capture)
    h = hits.iloc[0]
    expect = 1234 * 22.5 + (22.5 - 77 * 60.0 / 255)
    assert h.timestamp_ns == pytest.approx(expect)
    assert h.abs_time_ns == pytest.approx(1000 * 25.0 + expect)
    assert hits.hit_valid.all()


def _two_hit_packet(tmp_path):
    return write_pcapng(tmp_path / "l.pcapng", [srs_payload(
        1, 1, [hit_word(0, 1, 1, 1, 1), hit_word(0, 2, 2, 2, 2)])])


def test_every_word_of_packet_is_read(tmp_path):
    # a payload of 16 + 6*N bytes holds N words, as in vmm-sdat ParserSRS
    hits, meta = p2_decode.decode(_two_hit_packet(tmp_path))
    assert meta["n_words"] == 2
    assert hits.ch.tolist() == [1, 2]
    assert "legacy_word_count" not in meta


def test_legacy_word_count_skips_last_word(tmp_path):
    # parsers before 0.2 read range(0, len - 22, 6) and so yield N-1 words
    hits, meta = p2_decode.decode(_two_hit_packet(tmp_path),
                                  legacy_word_count=True)
    assert meta["n_words"] == 1 and meta["legacy_word_count"] is True
    assert hits.ch.tolist() == [1]


def test_single_word_packet(tmp_path):
    p = write_pcapng(tmp_path / "s.pcapng",
                     [srs_payload(1, 1, [hit_word(0, 5, 1, 1, 1)])])
    assert p2_decode.decode(p)[0].ch.tolist() == [5]
    assert len(p2_decode.decode(p, legacy_word_count=True)[0]) == 0


def test_chunk_invariance(capture):
    whole, _ = p2_decode.decode(capture)
    chunked, meta = p2_decode.decode(capture, max_words=1)
    assert meta["n_chunks"] == 2
    for c in p2_decode.COLUMNS:
        assert np.array_equal(whole[c].to_numpy(), chunked[c].to_numpy()), c


def test_store_round_trip(capture, tmp_path):
    whole, _ = p2_decode.decode(capture)
    meta = p2_decode.decode_to(capture, tmp_path / "store", max_words=1)
    assert meta["schema_version"] == p2_decode.SCHEMA_VERSION
    back, _ = p2_decode.load_frame(tmp_path / "store")
    for c in p2_decode.COLUMNS:
        assert np.array_equal(whole[c].to_numpy(), back[c].to_numpy()), c
        assert back[c].dtype == whole[c].dtype, c


def test_wrong_fec_ip_gives_no_hits(capture):
    hits, meta = p2_decode.decode(capture, src_ips={"10.0.0.9"})
    assert len(hits) == 0 and meta["n_hits"] == 0


def test_gray2bin_round_trip():
    n = np.arange(4096, dtype=np.uint16)
    assert np.array_equal(gray2bin(n ^ (n >> 1)), n)


def test_not_a_pcapng(tmp_path):
    p = tmp_path / "x.pcapng"
    p.write_bytes(b"\xd4\xc3\xb2\xa1" + b"\x00" * 20)
    with pytest.raises(ValueError, match="classic .pcap"):
        p2_decode.decode(p)


def test_capture_meta(capture):
    # builder: IDB without if_tsresol (microseconds), packet i stamped i us
    _, meta = p2_decode.decode(capture)
    c = meta["capture"]
    assert c["t_first_ns"] == 0.0 and c["t_last_ns"] == 1000.0
    assert c["live_time_s"] == pytest.approx(1e-6)
    assert c["frame_counter_changes"] is True          # frames 7 and 8
    assert c["per_fec"] == {"1": {"packets": 2, "udp_payload_bytes": 2 * (16 + 18),
                                  "hits": 4, "markers": 2}}
    assert c["udp_payload_bytes"] == 68


def test_nanosecond_timestamps(tmp_path):
    p = write_pcapng(tmp_path / "ns.pcapng",
                     [srs_payload(0, 2, [hit_word(0, 1, 1, 1, 1)])] * 3, tsresol=9)
    c = p2_decode.decode(p)[1]["capture"]
    assert c["live_time_s"] == pytest.approx(2e-9)
    assert c["frame_counter_changes"] is False


def test_decode_to_on_chunk(capture, tmp_path):
    seen = []
    meta = p2_decode.decode_to(capture, tmp_path / "s", max_words=1,
                               on_chunk=lambda c: seen.append(len(c["vmm"])))
    assert len(seen) == meta["n_chunks"] and sum(seen) == meta["n_hits"]
