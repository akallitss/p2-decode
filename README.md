# p2-decode

SRS/VMM3a raw data (pcapng) to hits, in electronics coordinates only:
`(fec, vmm, ch, adc, tdc, bcid, timestamps...)`. No detectors, connectors or
pads here — that is `p2-mapping`. Part of the P2 readout software split
(decode / mapping / monitor); see the design note.

## Install

```bash
pip install -e /path/to/p2-decode          # development checkout
pip install "p2-decode @ git+<repo-url>@v0.4.0"   # pinned, for production
```

Needs only numpy and pandas.

## Use

```python
from p2_decode import decode, decode_to, load_frame, iter_chunks

hits, meta = decode("capture.pcapng")              # whole capture, DataFrame
meta = decode_to("capture.pcapng", "store/cap1")   # streamed to a column store
hits, meta = load_frame("store/cap1")              # read it back
for cols in iter_chunks("capture.pcapng"): ...     # bounded memory

from p2_decode import StreamDecoder                # a capture still being written
sd = StreamDecoder("growing.pcapng")
cols, info = sd.poll()                             # hits since the last poll (or None)
sd.meta()                                          # cumulative, as decode() reports it
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
| `follow` | `PcapngFollower` (complete blocks appended since the last read), `StreamDecoder` (polls with marker state carried over) |
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
`P2_basket_online_analysis/vmm_decode.py` in SRS and TRG modes. It runs that comparison with `legacy_word_count=True` (see
below) and separately checks what the default adds.

## Versions

- **0.6.1** default chunk 500k words instead of 5M: a third of the memory
  (247 vs 734 MB peak for a full reduce of a 141 MB capture) and a little
  faster. Decoded values unchanged.
- **0.6.0** `StreamDecoder.skip_to_tail(keep_bytes)`: a monitor that falls
  behind can drop its backlog, walking block headers only; skipped bytes and
  packets are reported in `meta()["skipped"]`.
- **0.5.0** `decode_to(..., on_chunk=f)` calls `f` with each chunk's columns
  before writing them, so histograms can be built in the same pass.
- **0.4.0** decodes a capture while dumpcap writes it: `StreamDecoder.poll()`
  returns the hits that arrived since the last poll. The per-packet core is now
  `WordDecoder`, shared with the whole-file path; streaming a real capture in 40
  random byte pieces gives exactly the hits and meta of decoding it whole
  (`tests/test_follow.py`, SRS and TRG). With `src_ips=None` the stream keeps
  every VM3 source rather than those of the first 500 packets; for one FEC these
  agree.
- **0.3.1** the synthetic-capture builder used by the tests is now
  `p2_decode.testing`, so other packages' tests can use it.
- **0.3.0** adds `meta["capture"]`, for monitoring: `t_first_ns` / `t_last_ns`
  (pcapng capture times, ns since the epoch, honouring `if_tsresol`),
  `live_time_s`, `udp_payload_bytes`, and per FEC `packets`,
  `udp_payload_bytes`, `hits`, `markers`. `frame_counter_changes` says whether
  the SRS frame counter can count lost frames: the ALINX firmware sends a
  constant, so it cannot. Hits are unchanged.
- **0.2.0** reads every 6-byte word after the 16-byte SRS header, as vmm-sdat's
  `ParserSRS` does. The earlier Python parsers (`vmm_pcapng_qa.py`,
  `vmm_decode.py`, 0.1.0) read `range(0, len - 22, 6)` and skipped the last
  word of every packet. On four real captures this adds 0.07 % hits, leaves
  every previously decoded hit's raw fields identical, and corrects the
  `srs_timestamp` of about 0.08 % of hits that had been given a stale marker.
  `legacy_word_count=True` (CLI `--legacy-word-count`) reproduces the old
  output, for comparison with stores made by the old parsers.
- **0.1.0** is `vmm_decode.py` (P2_basket_online_analysis commit 812c3b9)
  split into modules with no change to any decoded value.
