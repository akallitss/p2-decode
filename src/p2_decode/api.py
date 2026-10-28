"""In-memory decode: a whole capture to one hits DataFrame."""

import numpy as np
import pandas as pd

from .calibration import derive
from .schema import COLUMNS, RAW_COLUMNS, empty_hits
from .srs import MAX_WORDS, iter_chunks


def decode(path, data_format="SRS", src_ips=None, max_packets=None,
           calibration=None, max_words=MAX_WORDS, legacy_word_count=False):
    """Decode a VMM pcapng into the hits DataFrame. Returns (hits, meta).

    calibration: optional 4-tuple (time_offset, time_slope, adc_offset,
    adc_slope), each (32, 64) float64, as returned by `load_calibration`.
    max_words: decode at most this many VMM words per chunk, bounding peak
    memory on multi-GB captures. Results are independent of the value.
    legacy_word_count: skip the last word of each packet, as the parsers
    before p2-decode 0.2 did (see `iter_chunks`).

    Holds every hit in memory. For multi-GB captures use `decode_to`, which
    streams to disk instead -- a full 2 GB run is ~3e8 hits, which is ~18 GB
    as a DataFrame.
    """
    meta = {}
    parts = list(iter_chunks(path, data_format, src_ips, max_packets,
                             calibration, max_words, meta, legacy_word_count))
    if not parts:
        return empty_hits(), meta

    cols = {}
    for name in RAW_COLUMNS:
        cols[name] = (np.concatenate([p[name] for p in parts]) if len(parts) > 1
                      else parts[0][name])
        for p in parts:
            del p[name]
    del parts
    derive(cols, calibration)
    return pd.DataFrame({c: cols[c] for c in COLUMNS}), meta
