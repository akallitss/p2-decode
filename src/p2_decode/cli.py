"""`p2-decode` command: decode one capture (or a port) and print its summary.

    p2-decode FILE [--format SRS|TRG] [--out-dir DIR] [--max-packets N]
    p2-decode --file FILE [--npackets N] [--duration S] [--out-dir DIR]
    p2-decode --port [HOST:]PORT (--npackets N | --duration S) [--out-dir DIR]

--npackets is the same option as --max-packets. With a plain FILE and no
--duration the whole-file path (decode / decode_to) is used, as before; with
--duration or --port the packets are read in bounded batches (HitSource).
"""

import argparse
import os
import time

from .api import decode
from .schema import COLUMNS, RAW_COLUMNS
from .srs import MAX_WORDS
from .store import HitWriter, decode_to


def _from_source(args):
    """--port / --duration: HitSource batches, to memory or a column store."""
    import pandas as pd

    from .source import HitSource, _frame

    src = HitSource(file=args.file, port=args.port, data_format=args.format,
                    legacy_word_count=args.legacy_word_count, max_words=args.max_words)
    writer = HitWriter(args.out_dir, columns=RAW_COLUMNS) if args.out_dir else None
    parts, left = [], args.max_packets
    try:
        while True:
            n0 = src.n_packets
            cols, packets = src.get_cols(npackets=left, duration=args.duration)
            if left is not None:
                left -= src.n_packets - n0
            if cols is not None:
                if writer is not None:
                    writer.append(cols)
                else:
                    parts.append(_frame(cols))
            # one call meets --duration (or the end of the file / the timeout);
            # only --npackets on a port can need several
            if args.duration is not None or not len(packets) or not left:
                break
    finally:
        src.close()
    meta = src.meta()
    meta["source"] = (f"udp://{args.port}" if args.port
                      else os.path.abspath(args.file))
    if writer is not None:
        from . import __version__
        from .schema import DERIVED_COLUMNS, SCHEMA_VERSION
        meta.update(columns=list(RAW_COLUMNS), derived_columns=list(DERIVED_COLUMNS),
                    constant_columns={}, schema_version=SCHEMA_VERSION,
                    p2_decode_version=__version__)
        writer.close(meta)
        return None, meta
    hits = (pd.concat(parts, ignore_index=True) if parts
            else _frame(None))[COLUMNS]
    return hits, meta


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="p2-decode",
        description="Decode an SRS/VMM3a pcapng (or the FEC's UDP stream) into hits.")
    ap.add_argument("pcap_file", nargs="?")
    ap.add_argument("--file", help="the capture (same as the positional argument)")
    ap.add_argument("--port", metavar="[HOST:]PORT",
                    help="read the FEC's UDP stream on this port instead of a file")
    ap.add_argument("--format", choices=["SRS", "TRG"], default="SRS")
    ap.add_argument("--max-packets", "--npackets", dest="max_packets", type=int,
                    default=None, metavar="N")
    ap.add_argument("--duration", type=float, metavar="S",
                    help="decode S seconds of data (packet time)")
    ap.add_argument("--out-dir", metavar="DIR",
                    help="stream hits to a column store here instead of "
                         "holding them in memory (required for multi-GB files)")
    ap.add_argument("--max-words", type=int, default=MAX_WORDS,
                    help=f"VMM words decoded per chunk (default {MAX_WORDS:,})")
    ap.add_argument("--legacy-word-count", action="store_true",
                    help="skip the last word of each packet, as parsers "
                         "before p2-decode 0.2 did (for comparisons only)")
    args = ap.parse_args(argv)
    if args.pcap_file and args.file and args.pcap_file != args.file:
        ap.error("give the capture once (positional or --file)")
    args.file = args.file or args.pcap_file
    if (args.file is None) == (args.port is None):
        ap.error("give a capture file or --port")
    if args.port and args.max_packets is None and args.duration is None:
        ap.error("--port needs --npackets or --duration")

    t0 = time.time()
    if args.port or args.duration is not None:
        hits, meta = _from_source(args)
    elif args.out_dir:
        meta = decode_to(args.file, args.out_dir, data_format=args.format,
                         max_packets=args.max_packets, max_words=args.max_words,
                         legacy_word_count=args.legacy_word_count)
        hits = None
    else:
        hits, meta = decode(args.file, data_format=args.format,
                            max_packets=args.max_packets,
                            max_words=args.max_words,
                            legacy_word_count=args.legacy_word_count)
    dt = time.time() - t0
    print(f"udp://{args.port}" if args.port else f"{args.file}")
    for k, v in meta.items():
        print(f"  {k:18s} {v}")
    print(f"  {'decode_s':18s} {dt:.2f}")
    if args.out_dir:
        tot = sum(os.path.getsize(os.path.join(args.out_dir, f))
                  for f in os.listdir(args.out_dir))
        print(f"  {'store_MB':18s} {tot/1e6:.1f}  ({args.out_dir})")
    else:
        print(hits.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
