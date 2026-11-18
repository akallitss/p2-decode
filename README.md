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

Bounded batches, from a file or straight from the FEC's UDP port (0.9):

```python
from p2_decode import get_packets, get_hits, decode, HitSource

packets = get_packets(file="cap.pcapng", npackets=1000)   # first 1000 VM3 packets
hits, meta = decode(packets)                              # ... decoded
hits = get_hits(file="cap.pcapng", duration=2.0)          # first 2 s of data time
hits = get_hits(port=6006, duration=5.0)                  # 5 s off the network

src = HitSource(file="cap.pcapng")      # or port="192.168.1.16:6006", or follow=True
a = src.get(npackets=1000)              # carries on from the last batch,
b = src.get(duration=1.0)               # marker state kept: same hits as one decode()
```

`npackets` counts VM3 packets; `duration` is seconds of packet (data) time. With
both, the first reached wins. Markers are not returned as rows: each one sets
`srs_timestamp` / `trigger_time` on the following hits of its VMM.

```bash
p2-decode capture.pcapng [--format SRS|TRG] [--out-dir DIR] [--max-packets N]
p2-decode --file capture.pcapng --npackets N | --duration S [--out-dir DIR]
p2-decode --port [HOST:]PORT --npackets N | --duration S [--out-dir DIR]
```

## Layers

| Module | Does |
| --- | --- |
| `pcapng` | pcapng blocks, Ethernet/VLAN/IPv4/UDP, vectorised over packets |
| `srs` | SRS header, 6-byte VMM words, marker state machine, FEC IP detection |
| `calibration` | derived columns (timing, calibrated ADC), vmm-sdat calibration loader |
| `api`, `store` | in-memory decode; streaming `.npy` column store |
| `source` | `get_packets`, `get_hits`, `PacketSource` / `HitSource`: bounded batches (npackets, duration) from a file, a growing file or a UDP port |
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
pytest                                  # synthetic captures + the small real one, no data disk needed
P2_DECODE_LEGACY=.../vmm_decode.py \
P2_DECODE_CAPTURES=a.pcapng:b.pcapng pytest tests/test_legacy.py
```

`test_legacy.py` checks bit-identity, column by column, with the original
`P2_basket_online_analysis/vmm_decode.py` in SRS and TRG modes. It runs that comparison with `legacy_word_count=True` (see
below) and separately checks what the default adds.

`test_reference.py` decodes a small real capture, `tests/data/lab_run263_head.pcapng`
(33 packets of the lab cosmic stand, 298 kB), through every entry point -- whole
file, column store, batches, replayed through a UDP port -- and compares with the
reference hits next to it. That reference passed `test_legacy.py` when it was
made; `tests/data/make_reference.py` says how, and regenerates it.

CI: `.github/workflows/tests.yml` runs `pytest` on every push (Python 3.9, 3.12,
3.13); `.gitlab-ci.yml` does the same on the CEA GitLab copy, except on the
mirrored `ak/*` branches.

## Versions

- **0.9.0** `get_packets` / `get_hits` (file or port, `npackets=`, `duration=`),
  `PacketSource` / `HitSource` to carry on batch after batch, `decode(packets)`;
  `p2-decode --port`, `--duration`, `--file`, `--npackets` (= `--max-packets`).
  A small real capture with reference hits under `tests/data`; CI on GitHub
  and GitLab. Nothing existing changed: `decode(path)`, `iter_chunks`, `derive`,
  the store and the existing CLI options behave as in 0.8.1.

- **0.8.1** decoded chunks (iter_chunks, StreamDecoder, UdpStreamDecoder) carry
  `pkt_time_ns`, the packet's capture/receive time per hit, for rate-vs-time;
  not a stored column. Legacy test ignores `n_chunks` (depends on MAX_WORDS).
- **0.8.0** `UdpStreamDecoder("192.168.1.16:6006")`: decode the FEC's UDP stream
  directly, no file. A plain UDP socket on the FEC's destination port: the
  kernel gives every datagram to dumpcap's capture and to this socket, so the
  recording is never affected. Datagrams this reader is too slow for are
  dropped by the kernel for this socket only and counted (SO_RXQ_OVFL), so
  `fraction_received` scales rates to the full stream. Kernel receive
  timestamps (SO_TIMESTAMPNS) give the live time. Same WordDecoder as the
  file paths: identical hits (tests/test_udp.py). Linux only, no privileges.
- **0.7.0** `decode_to(..., columns=[...])` stores only those raw columns.
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
