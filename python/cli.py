"""`p2-decode` command: decode one capture and print its summary."""

import argparse
import os
import time

from .api import decode
from .srs import MAX_WORDS
from .store import decode_to


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="p2-decode",
        description="Decode an SRS/VMM3a pcapng into hits.")
    ap.add_argument("pcap_file")
    ap.add_argument("--format", choices=["SRS", "TRG"], default="SRS")
    ap.add_argument("--max-packets", type=int, default=None)
    ap.add_argument("--out-dir", metavar="DIR",
                    help="stream hits to a column store here instead of "
                         "holding them in memory (required for multi-GB files)")
    ap.add_argument("--max-words", type=int, default=MAX_WORDS,
                    help=f"VMM words decoded per chunk (default {MAX_WORDS:,})")
    ap.add_argument("--legacy-word-count", action="store_true",
                    help="skip the last word of each packet, as parsers "
                         "before p2-decode 0.2 did (for comparisons only)")
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.out_dir:
        meta = decode_to(args.pcap_file, args.out_dir, data_format=args.format,
                         max_packets=args.max_packets, max_words=args.max_words,
                         legacy_word_count=args.legacy_word_count)
        hits = None
    else:
        hits, meta = decode(args.pcap_file, data_format=args.format,
                            max_packets=args.max_packets,
                            max_words=args.max_words,
                            legacy_word_count=args.legacy_word_count)
    dt = time.time() - t0
    print(f"{args.pcap_file}")
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
