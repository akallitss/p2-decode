"""The hits table: column names, dtypes and the timing constants behind them.

This is the contract every consumer of p2-decode relies on. Any change to the
columns, their dtypes or their meaning bumps SCHEMA_VERSION, which is written
into every column store's meta.json.
"""

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

# the hits table, in order. dtypes are fixed by the reference parser.
COLUMNS = [
    "fec", "vmm", "time", "ch", "adc", "adc_calibrated", "over_threshold",
    "offset", "bcid", "tdc", "timestamp_ns", "srs_timestamp", "abs_time_ns",
    "trigger_time", "trigger_counter", "hit_valid",
]

# What the hardware actually sends. These are stored.
RAW_COLUMNS = [
    "fec", "vmm", "time", "ch", "adc", "over_threshold", "offset", "bcid",
    "tdc", "srs_timestamp", "trigger_time", "trigger_counter",
]

# Pure functions of RAW_COLUMNS (plus a calibration). Recomputed on load rather
# than stored: they are 27 MB per 11 MB capture, and storing them would freeze
# one calibration into the file, forcing a re-decode whenever it is retuned.
DERIVED_COLUMNS = ["adc_calibrated", "timestamp_ns", "abs_time_ns", "hit_valid"]

COLUMN_DTYPES = {
    "fec": np.uint8, "vmm": np.uint8, "time": np.uint32, "ch": np.uint8,
    "adc": np.uint16, "adc_calibrated": np.float32, "over_threshold": np.bool_,
    "offset": np.int8, "bcid": np.uint16, "tdc": np.uint8,
    "timestamp_ns": np.float64, "srs_timestamp": np.uint64,
    "abs_time_ns": np.float64, "trigger_time": np.uint64,
    "trigger_counter": np.uint16, "hit_valid": np.bool_,
}

# --- physics/format constants ---
CLOCK_PERIOD_NS = 22.5   # ns per BCID count (44.44 MHz clock)
TAC_SLOPE_NS = 60.0      # ns full-scale of the TDC TAC ramp
TDC_RANGE = 255          # TDC full-scale bin count (vmm-sdat SRSTime::tdc_range)


def empty_hits():
    """A zero-row hits DataFrame with the full schema."""
    return pd.DataFrame({c: np.zeros(0, COLUMN_DTYPES[c]) for c in COLUMNS})
