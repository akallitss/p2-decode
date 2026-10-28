"""Bit-identity against the original P2_basket_online_analysis/vmm_decode.py.

The original skips the last word of every packet, so the comparison runs with
legacy_word_count=True; test_full_word_count checks what the default adds.

Runs only when both are available:
    P2_DECODE_LEGACY   path to the legacy vmm_decode.py
    P2_DECODE_CAPTURES colon-separated list of real .pcapng captures
"""

import importlib.util
import os

import numpy as np
import pytest

import p2_decode

LEGACY = os.environ.get("P2_DECODE_LEGACY")
CAPTURES = [p for p in os.environ.get("P2_DECODE_CAPTURES", "").split(":") if p]

pytestmark = pytest.mark.skipif(
    not (LEGACY and CAPTURES), reason="set P2_DECODE_LEGACY and P2_DECODE_CAPTURES")


@pytest.fixture(scope="module")
def legacy():
    spec = importlib.util.spec_from_file_location("legacy_vmm_decode", LEGACY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("data_format", ["SRS", "TRG"])
@pytest.mark.parametrize("path", CAPTURES, ids=os.path.basename)
def test_identical_to_legacy(legacy, path, data_format):
    ref, ref_meta = legacy.decode(path, data_format=data_format)
    new, new_meta = p2_decode.decode(path, data_format=data_format,
                                     legacy_word_count=True)
    assert new_meta.pop("legacy_word_count") is True
    new_meta.pop("capture")              # added in 0.3; the legacy decoder has none
    # how many chunks the decode took depends on MAX_WORDS (500k since 0.6.1,
    # the legacy decoder used 5M); the hits do not, as the loop below checks
    new_meta.pop("n_chunks")
    ref_meta.pop("n_chunks")
    assert len(new) == len(ref) > 0
    assert list(new.columns) == list(ref.columns)
    for c in ref.columns:
        a, b = ref[c].to_numpy(), new[c].to_numpy()
        assert a.dtype == b.dtype, c
        assert np.array_equal(a, b), c
    assert new_meta == ref_meta


@pytest.mark.parametrize("path", CAPTURES, ids=os.path.basename)
def test_full_word_count(legacy, path):
    _, ref_meta = legacy.decode(path)
    _, meta = p2_decode.decode(path)
    # exactly one more word per VM3 packet: every DAQ frame is 16 + 6*N bytes
    n_vm3_frames = meta["n_words"] - ref_meta["n_words"]
    assert 0 < n_vm3_frames <= meta["n_packets"]
    assert meta["n_hits"] + meta["n_markers"] == meta["n_words"]
    assert meta["n_hits"] >= ref_meta["n_hits"]
