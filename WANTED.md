
# p2basket-decode

## Scope

Just decode standard SRS pcapng data for now.
Will evolve when readout out tracks, etc.

Think again about functionality, then remove any unneeded files or functions

## Desired python features (for discussion! but noted here)

### just get packets, from given file or port. Bound by time, nb of packets, other?

```python
from p2basket-decode import blabla

packets = get_packets(file=FILE)                    # indefinite, or with some default boundary
packets = get_packets(file=FILE, npackets=NUMBER)   # get NUMBER packets, starting from the last fetched packet
packets = get_packets(file=FILE, duration=DURATION) # get packets for DURATION seconds

packets = get_packets(port=PORT)                    # indefinite, or with some default boundary
packets = get_packets(port=PORT, npackets=NUMBER)   # get NUMBER packets, starting from the last fetched packet
packets = get_packets(port=PORT, duration=DURATION) # get packets for DURATION seconds
```

### decode given packet(s) to hit(s) or marker(s)

```python
hit  = decode(packet)
hits = decode(packets)
```

Not sure : how to handle hits vs markers?

### file or port to hits in one go (get_hits uses get_packets and decode)

```python
from p2basket-decode import blabla

hits = get_hits(file=FILE)                          # indefinite, or with some default boundary
hits = get_hits(file=FILE, npackets=NUMBER)         # get NUMBER hits, starting from the last fetched packet
hits = get_hits(file=FILE, duration=DURATION)       # get hits for DURATION seconds

hits = get_hits(port=PORT)                          # indefinite, or with some default boundary
hits = get_hits(port=PORT, npackets=NUMBER)         # get NUMBER hits, starting from the last fetched packet
hits = get_hits(port=PORT, duration=DURATION)       # get hits for DURATION seconds
```

## Command line execution

```bash
p2-decode --file FILE [--format SRS|TRG] [--out-dir DIR] [--npackets N] [--duration N]
p2-decode --port PORT [--format SRS|TRG] [--out-dir DIR] [--npackets N] [--duration N]
```

## Tests (to be used in CI)

- analyse a small test pcapng file and compare to reference
- push this file through a port, listen there, decode, and compare to reference