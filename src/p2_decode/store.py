"""Streaming column store: one .npy per raw column plus meta.json.

Chosen over parquet/HDF5 because the DAQ venv has none of pyarrow, fastparquet,
h5py or pytables, and adding a dependency to the machine the beam depends on is
not worth it. The format costs nothing: readers open columns with
np.load(mmap_mode='r') and page in only what they touch, so a per-VMM ADC
histogram never reads the timestamp columns at all.

.npy stores its length in a header, which is not known until the last chunk
is written. The header is therefore written with shape (0,) and a fixed
padded length, then rewritten in place at close.
"""

import json
import os
import struct

import numpy as np
import pandas as pd

from . import __version__
from .calibration import derive
from .schema import (COLUMNS, COLUMN_DTYPES, DERIVED_COLUMNS, RAW_COLUMNS,
                     SCHEMA_VERSION)
from .srs import MAX_WORDS, iter_chunks

_NPY_HDR_LEN = 118          # 10-byte prefix + 118 = 128, a multiple of 64


def _npy_header(dtype, n):
    from numpy.lib import format as npf
    descr = npf.dtype_to_descr(np.dtype(dtype))
    s = "{'descr': %r, 'fortran_order': False, 'shape': (%d,), }" % (descr, n)
    pad = _NPY_HDR_LEN - len(s) - 1
    if pad < 0:
        raise ValueError(f"npy header too long for {dtype!r}")
    body = (s + " " * pad + "\n").encode("latin1")
    return b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", _NPY_HDR_LEN) + body


class HitWriter:
    """Append hit columns to a directory of .npy files without buffering them."""

    def __init__(self, out_dir, columns=RAW_COLUMNS, dtypes=COLUMN_DTYPES,
                 constants=None):
        self.dir = out_dir
        self.columns = list(columns)
        self.dtypes = dict(dtypes)
        # Columns known to hold one value for every hit are recorded in
        # meta.json instead of being written out. In SRS mode trigger_time and
        # trigger_counter are provably 0 -- only the TRG marker branch ever
        # writes them -- and they are 10 of the 32 bytes per hit.
        self.constants = dict(constants or {})
        self.stored = [c for c in self.columns if c not in self.constants]
        os.makedirs(out_dir, exist_ok=True)
        self.n = 0
        self._f = {}
        for c in self.stored:
            f = open(os.path.join(out_dir, f"{c}.npy"), "wb")
            f.write(_npy_header(self.dtypes[c], 0))
            self._f[c] = f

    def append(self, cols):
        n_new = None
        for c in self.columns:
            a = np.ascontiguousarray(cols[c], dtype=self.dtypes[c])
            if n_new is None:
                n_new = len(a)
            elif len(a) != n_new:
                raise ValueError(f"column {c} has {len(a)} rows, expected {n_new}")
            if c in self.constants:
                v = self.constants[c]
                if a.size and not bool((a == v).all()):
                    raise ValueError(
                        f"column {c} was declared constant {v!r} but this chunk "
                        "contains other values")
                continue
            self._f[c].write(a.tobytes())
        self.n += n_new or 0

    def close(self, meta=None):
        for c in self.stored:
            f = self._f[c]
            f.flush()
            f.seek(0)
            f.write(_npy_header(self.dtypes[c], self.n))   # same length, in place
            f.close()
        self._f.clear()
        if meta is not None:
            with open(os.path.join(self.dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2, sort_keys=True, default=str)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._f:
            self.close()


def decode_to(path, out_dir, data_format="SRS", src_ips=None, max_packets=None,
              calibration=None, max_words=MAX_WORDS, legacy_word_count=False,
              on_chunk=None):
    """Decode a pcapng straight to a column store, at flat memory cost.

    Peak memory is one chunk, not one file, so a 2 GB capture (~3e8 hits, which
    is ~18 GB as a DataFrame) decodes in bounded RAM. Returns the meta dict,
    which is also written to <out_dir>/meta.json.

    on_chunk: called with each chunk's raw columns before they are written, so
    a caller can accumulate histograms in the same pass instead of reading
    the whole store back.
    """
    meta = {}
    # In SRS mode the trigger slots are never written by the marker decoder,
    # so they are constant 0 and need not hit the disk (31 % of the store).
    constants = ({"trigger_time": 0, "trigger_counter": 0}
                 if data_format == "SRS" else {})
    writer = HitWriter(out_dir, constants=constants)
    try:
        for cols in iter_chunks(path, data_format, src_ips, max_packets,
                                calibration, max_words, meta,
                                legacy_word_count):
            if on_chunk is not None:
                on_chunk(cols)
            writer.append(cols)
            del cols
    except BaseException:
        writer.close()
        raise
    meta["source"] = os.path.abspath(path)
    meta["columns"] = list(RAW_COLUMNS)
    meta["derived_columns"] = list(DERIVED_COLUMNS)
    meta["constant_columns"] = {k: int(v) for k, v in constants.items()}
    meta["schema_version"] = SCHEMA_VERSION
    meta["p2_decode_version"] = __version__
    writer.close(meta)
    return meta


def load_hits(out_dir, columns=None, mmap=True, derive_cols=True,
              calibration=None):
    """Read a column store back. Returns (dict of arrays, meta).

    Memory-mapped by default, so reading one column of a 2 GB store pages in
    only that column. `derive_cols` recomputes the derived columns via
    `derive()`; that materialises float64 arrays, so pass False (or a narrow
    `columns` list) when you only need raw ones.
    """
    meta_path = os.path.join(out_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    stored = meta.get("columns", RAW_COLUMNS)

    if columns is None:
        names = list(stored)
    else:
        # pull in whatever raw columns the requested derived ones need
        names = [c for c in columns if c in stored]
        if derive_cols and any(c in DERIVED_COLUMNS for c in columns):
            names = list(dict.fromkeys(names + list(stored)))

    const = meta.get("constant_columns", {})
    n = int(meta.get("n_hits", 0))
    out = {}
    for c in names:
        if c in const:
            out[c] = np.full(n, const[c], dtype=COLUMN_DTYPES[c])
        else:
            out[c] = np.load(os.path.join(out_dir, f"{c}.npy"),
                             mmap_mode="r" if mmap else None)
    if derive_cols and set(RAW_COLUMNS) <= set(out):
        derive(out, calibration)
    if columns is not None:
        out = {c: out[c] for c in columns if c in out}
    return out, meta


def load_frame(out_dir, calibration=None):
    """The column store as the same 16-column DataFrame `decode` returns."""
    cols, meta = load_hits(out_dir, calibration=calibration, mmap=False)
    return pd.DataFrame({c: cols[c] for c in COLUMNS}), meta
