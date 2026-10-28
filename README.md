# p2-decode

SRS/VMM3a raw data (pcapng) to hits, in electronics coordinates only:
`(fec, vmm, ch, adc, tdc, bcid, timestamps...)`. No detectors, connectors or
pads here — that is `p2-mapping`. Part of the P2 readout software split
(decode / mapping / monitor); see the design note.

## Install

```bash
pip install -e /path/to/p2-decode          # development checkout
pip install "p2-decode @ git+<repo-url>@v0.1.0"   # pinned, for production
```

Needs only numpy and pandas.

## Use

```python
from p2_decode import decode, decode_to, load_frame, iter_chunks

hits, meta = decode("capture.pcapng")              # whole capture, DataFrame
meta = decode_to("capture.pcapng", "store/cap1")   # streamed to a column store
hits, meta = load_frame("store/cap1")              # read it back
for cols in iter_chunks("capture.pcapng"): ...     # bounded memory
```

```bash
p2-decode capture.pcapng [--format SRS|TRG] [--out-dir DIR] [--max-packets N]
```

## Layers

| Module | Does |
| --- | --- |
| `pcapng` | pcapng blocks, Ethernet/VLAN/IPv4/UDP, vectorised over packets |
| `srs` | SRS header, 6-byte VMM words, marker state machine, FEC IP detection |
| `calibration` | derived columns (timing, calibrated ADC), vmm-sdat calibration loader |
| `api`, `store` | in-memory decode; streaming `.npy` column store |
| `schema` | column names, dtypes, timing constants, `SCHEMA_VERSION` |

## Hits schema (v1)

Stored: `fec vmm ch` (uint8), `adc` (uint16), `tdc` (uint8), `bcid` (uint16),
`time` (uint32, SRS frame counter), `offset` (int8), `over_threshold` (bool),
`srs_timestamp trigger_time` (uint64), `trigger_counter` (uint16).
Derived on load: `timestamp_ns abs_time_ns` (float64), `adc_calibrated`
(float32), `hit_valid` (bool).

## Tests

```bash
pytest                                  # synthetic captures, no data needed
P2_DECODE_LEGACY=.../vmm_decode.py \
P2_DECODE_CAPTURES=a.pcapng:b.pcapng pytest tests/test_legacy.py
```

`test_legacy.py` checks bit-identity, column by column, with the original
`P2_basket_online_analysis/vmm_decode.py` in SRS and TRG modes.

## Origin

v0.1.0 is `vmm_decode.py` (P2_basket_online_analysis commit 812c3b9) split into
modules with no change to any decoded value.
