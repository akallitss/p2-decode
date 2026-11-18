"""Regenerate the reference outputs of the test capture (run from the repo root).

    python tests/data/make_reference.py

Only do this for an intended change of the decoded values, and say why in the
commit: the reference is what test_reference.py holds p2-decode to.

How the reference was checked when it was made (2026-11-18, p2-decode 0.9.0):
  * lab_run263_head.pcapng is the first 33 packets (298 kB, 0.33 s) of
    run_263/cfg_gain6.0_..._hvon50_hvm/raw_daq_data/enp4s0f1_00001_20261118171801.pcapng
    on vmm_daplxa (lab cosmic stand, FEC 192.168.1.13, 19 VMMs), cut on whole
    pcapng blocks;
  * test_legacy.py passed on it against the original
    P2_basket_online_analysis/vmm_decode.py (bit-identical in legacy word-count
    mode), and the default mode adds exactly the last word of each packet.
"""

import json
import os

import numpy as np

import p2_decode

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE = os.path.join(HERE, "lab_run263_head.pcapng")


def main():
    hits, meta = p2_decode.decode(CAPTURE)
    np.savez_compressed(os.path.join(HERE, "lab_run263_head_hits.npz"),
                        **{c: hits[c].to_numpy() for c in p2_decode.RAW_COLUMNS})
    keep = {k: meta[k] for k in ("n_packets", "n_vm3_packets", "n_hits", "n_markers",
                                 "n_words", "n_invalid_offset", "fec_ips", "data_format")}
    keep["live_time_s"] = meta["capture"]["live_time_s"]
    with open(os.path.join(HERE, "lab_run263_head_meta.json"), "w") as f:
        json.dump(keep, f, indent=1, sort_keys=True)
        f.write("\n")
    print(keep)


if __name__ == "__main__":
    main()
