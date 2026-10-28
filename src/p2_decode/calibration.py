"""Derived columns (timing, calibrated ADC) and the vmm-sdat calibration loader."""

import json
import logging

import numpy as np

from .schema import CLOCK_PERIOD_NS, TAC_SLOPE_NS, TDC_RANGE

log = logging.getLogger(__name__)


def derive(cols, calibration=None):
    """Add the derived columns in place, from the raw ones.

    The single source of these formulas -- `decode` and `load_hits` both route
    through here, so an in-memory decode and a round-trip through the column
    store cannot drift apart.

    calibration: optional 4-tuple (time_offset, time_slope, adc_offset,
    adc_slope), each (32, 64) float64, as returned by `load_calibration`.
    """
    vmm, ch, adc = cols["vmm"], cols["ch"], cols["adc"]
    offset, bcid, tdc = cols["offset"], cols["bcid"], cols["tdc"]
    srs_ts = cols["srs_timestamp"]

    if calibration is None:
        t_off, t_slp, a_off, a_slp = 0.0, 1.0, 0.0, 1.0
    else:
        cal_to, cal_ts, cal_ao, cal_as = calibration
        vi = np.asarray(vmm, dtype=np.intp)
        ci = np.asarray(ch, dtype=np.intp)
        t_off, t_slp = cal_to[vi, ci], cal_ts[vi, ci]
        a_off, a_slp = cal_ao[vi, ci], cal_as[vi, ci]

    t_coarse = (np.asarray(offset, dtype=np.float64) * 4096 +
                np.asarray(bcid, dtype=np.float64)) * CLOCK_PERIOD_NS
    t_fine = (CLOCK_PERIOD_NS -
              np.asarray(tdc, dtype=np.float64) * TAC_SLOPE_NS / TDC_RANGE
              - t_off) * t_slp
    timestamp_ns = t_coarse + t_fine

    cols["adc_calibrated"] = (a_slp * np.asarray(adc, dtype=np.float64)
                              + a_off).astype(np.float32)
    cols["timestamp_ns"] = timestamp_ns
    cols["abs_time_ns"] = np.asarray(srs_ts, dtype=np.float64) * 25.0 + timestamp_ns
    cols["hit_valid"] = np.asarray(offset) >= np.int8(-1)
    return cols


def load_calibration(json_path):
    """Parse a vmm-sdat calibration JSON into per-(vmm, ch) correction arrays.

    Supports the vmm-sdat file format (vmmID + hybridID, arrays of 64 floats)
    and the C++ ParserSRS format (fecID + vmmID). fecID is ignored: calibration
    is applied by vmmID only, which is correct for single-FEC setups.

    Returns four numpy arrays shaped (32, 64):
        time_offset  -- subtracted from the TAC term before applying time_slope (ns)
        time_slope   -- multiplies (bc_factor - tdc_corr - time_offset)
        adc_offset   -- applied as:  adc_calibrated = adc_slope * adc + adc_offset
        adc_slope    -- (see above)

    Timewalk coefficients (timewalk_a/b/c/d) are recognised but not applied,
    since the application formula is not part of the base ParserSRS.
    """
    with open(json_path) as f:
        data = json.load(f)

    time_offset = np.zeros((32, 64), dtype=np.float64)
    time_slope = np.ones((32, 64), dtype=np.float64)
    adc_offset = np.zeros((32, 64), dtype=np.float64)
    adc_slope = np.ones((32, 64), dtype=np.float64)

    has_time, has_adc, has_tw = False, False, False
    for entry in data.get("vmm_calibration", []):
        vid = int(entry["vmmID"])
        if vid >= 32:
            continue

        def _load(key, arr, row):
            vals = entry.get(key, [])
            if vals:
                arr[row, :min(len(vals), 64)] = vals[:64]
        _load("time_offsets", time_offset, vid)
        has_time = has_time or bool(entry.get("time_offsets"))
        _load("time_slopes", time_slope, vid)
        _load("adc_offsets", adc_offset, vid)
        has_adc = has_adc or bool(entry.get("adc_offsets"))
        _load("adc_slopes", adc_slope, vid)
        if any(entry.get(k) for k in ("timewalk_a", "timewalk_b",
                                      "timewalk_c", "timewalk_d")):
            has_tw = True

    loaded = []
    if has_time:
        loaded.append("time_offset/slope")
    if has_adc:
        loaded.append("adc_offset/slope")
    if has_tw:
        loaded.append("timewalk (loaded, not applied)")
    log.info("calibration arrays loaded: %s",
             ", ".join(loaded) if loaded else "none found")
    return time_offset, time_slope, adc_offset, adc_slope
