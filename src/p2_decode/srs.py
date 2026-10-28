"""SRS/VMM3a frames and words: UDP payloads to raw hit columns.

Every 6-byte VMM data word in the file is decoded in a single set of numpy
expressions rather than one `struct.unpack_from` per word.

The one genuinely sequential piece is the marker state machine: each hit
carries the most recent marker values for its own (fec, vmm), so the value is
a forward-fill along the global word order. That is done per (fec, vmm) key
with `searchsorted` -- see `_last_value_before`.
"""

import numpy as np

from .pcapng import ip_str, ip_u32, read_packets

PROBE_PACKETS = 500      # packets sampled by detect_fec_ips

_VM3 = (ord("V") << 16) | (ord("M") << 8) | ord("3")

# marker state is keyed by (fec_id, vmm_id) with fec 0-15 and vmm 0-31
_NKEYS = 16 * 32

# words decoded per chunk; ~6 B raw + ~24 B of decoded columns each, so 5e6
# keeps a chunk's transient footprint around 150 MB
MAX_WORDS = 5_000_000


def detect_fec_ips(path, n_probe=PROBE_PACKETS):
    """Source IPs carrying VM3 payloads, sampled over the first n_probe packets."""
    u8, pstart, plen, src, _ = read_packets(path, max_packets=n_probe)
    if len(pstart) == 0:
        return {}
    m = plen > 6
    ps, sr = pstart[m], src[m]
    if len(ps) == 0:
        return {}
    magic = (u8[ps + 4].astype(np.uint32) << 16 |
             u8[ps + 5].astype(np.uint32) << 8 |
             u8[ps + 6].astype(np.uint32))
    sr = sr[magic == _VM3]
    vals, counts = np.unique(sr, return_counts=True)
    return {ip_str(int(v)): int(c) for v, c in zip(vals, counts)}


# ---------------------------------------------------------------------------
# marker forward-fill
# ---------------------------------------------------------------------------

def _last_value_before(upd_pos, upd_key, upd_val, q_pos, q_key, n_hits):
    """For each query, the most recent update value with the same key.

    This is the vectorised form of the reference parser's `markers` dict: a hit
    takes whatever was last written to markers[(fec, vmm)] before it in the
    word stream, or 0 if that key was never written.

    Both sides are collapsed to a single sort key `key * STRIDE + pos`, so one
    global searchsorted answers every hit at once. Because `pos` is bounded by
    STRIDE, ordering by the combined key orders first by key and then by
    position, which is exactly the "previous marker for my own (fec, vmm)"
    lookup. A per-key Python loop with boolean masks costs ~5 passes over the
    full hit array per key; this is one O(n log m) pass in total.
    """
    out = np.zeros(n_hits, dtype=np.uint64)
    if len(upd_pos) == 0 or n_hits == 0:
        return out

    stride = np.int64(max(int(q_pos[-1]) if n_hits else 0,
                          int(upd_pos[-1])) + 2)
    u_comb = upd_key.astype(np.int64) * stride + upd_pos
    order = np.argsort(u_comb, kind="stable")
    u_comb = u_comb[order]
    u_key = upd_key.astype(np.int64)[order]
    u_val = upd_val[order]

    q_comb = q_key.astype(np.int64) * stride + q_pos
    j = np.searchsorted(u_comb, q_comb, side="left") - 1
    # j < 0, or the update found belongs to a different key -> never written
    ok = j >= 0
    jc = np.clip(j, 0, None)
    ok &= u_key[jc] == q_key.astype(np.int64)
    np.copyto(out, u_val[jc], where=ok)
    return out


def _resolve(upd_pos, upd_key, upd_val, hit_pos, hit_key, n_hits, carry_row):
    """One marker slot: seed with the carry-in state, resolve, return new carry.

    The carry is injected as a synthetic update at position -1 for every key, so
    a hit that precedes any marker in this chunk still sees whatever the
    previous chunk left in markers[(fec, vmm)] -- exactly what the reference's
    persistent dict does. Keys never written anywhere carry 0, which is the
    reference's _DEFAULT_MARKER.
    """
    seed_key = np.arange(_NKEYS, dtype=np.int64)
    seed_pos = np.full(_NKEYS, -1, dtype=np.int64)
    all_pos = np.concatenate([seed_pos, upd_pos.astype(np.int64)])
    all_key = np.concatenate([seed_key, upd_key.astype(np.int64)])
    all_val = np.concatenate([carry_row, upd_val.astype(np.uint64)])

    out = _last_value_before(all_pos, all_key, all_val, hit_pos, hit_key, n_hits)

    # new carry = last update per key in this chunk, else the old carry
    new_carry = carry_row.copy()
    if len(upd_pos):
        k = upd_key.astype(np.int64)
        # updates arrive in position order, so the last occurrence of each key
        # is the first occurrence when scanned backwards
        ku, first_rev = np.unique(k[::-1], return_index=True)
        new_carry[ku] = upd_val.astype(np.uint64)[len(k) - 1 - first_rev]
    return out, new_carry


def _decode_markers(d1, d2, mrk_pos, fec_id, pkt_of_word, data_format,
                    hit_pos, hit_key, carry):
    """Resolve srs_timestamp / trigger_time / trigger_counter for every hit.

    `carry` is a (3, _NKEYS) uint64 array holding the marker state left by the
    previous chunk; the updated state is returned alongside the hit columns.
    """
    n_hits = len(hit_pos)
    empty_pos = np.zeros(0, dtype=np.int64)
    empty_val = np.zeros(0, dtype=np.uint64)

    if len(mrk_pos):
        m1 = d1[mrk_pos]
        m2 = d2[mrk_pos]
        m_fec = fec_id[pkt_of_word[mrk_pos]].astype(np.int64)
        vmmid = ((m2 >> 10) & 0x1F).astype(np.int64)
        # 42-bit value: d1 is 32 bits, so promote before shifting
        val42 = ((m1.astype(np.uint64) << np.uint64(10)) |
                 (m2 & 0x3FF).astype(np.uint64))
    else:
        m1 = m2 = m_fec = vmmid = val42 = None

    new_carry = np.empty_like(carry)

    if data_format == "SRS":
        # Reference takes vmmid at face value in SRS mode: this FEC hosts 20
        # VMMs, and gating on `vmmid < 16` silently dropped every marker for
        # VMMs 16-19.
        if len(mrk_pos):
            up, uk, uv = mrk_pos, m_fec * 32 + vmmid, val42
        else:
            up, uk, uv = empty_pos, empty_pos, empty_val
        srs, new_carry[0] = _resolve(up, uk, uv, hit_pos, hit_key, n_hits, carry[0])
        z64 = np.zeros(n_hits, dtype=np.uint64)
        new_carry[1] = carry[1]
        new_carry[2] = carry[2]
        return srs, z64, z64.copy(), new_carry

    # --- TRG ---
    if len(mrk_pos):
        is_vmm = vmmid < 16
        # VMM markers: only stored when the value fits in a BCID (< 4096)
        sel = is_vmm & (val42 < np.uint64(4096))
        trg = ~is_vmm
        trg_key = m_fec * 32 + (vmmid % 16)
        flag = (m1 >> 28) & 0x0F
        is_ctr = trg & (flag == 0xF)
        is_time = trg & (flag != 0xF)
        ctr_val = ((m1 & 0x3F).astype(np.uint64) * np.uint64(1024) +
                   (m2 & 0x3FF).astype(np.uint64))
        time_val = m1.astype(np.uint64) << np.uint64(10)
        srs_args = (mrk_pos[sel], (m_fec * 32 + vmmid)[sel], val42[sel])
        ctr_args = (mrk_pos[is_ctr], trg_key[is_ctr], ctr_val[is_ctr])
        tim_args = (mrk_pos[is_time], trg_key[is_time], time_val[is_time])
    else:
        srs_args = ctr_args = tim_args = (empty_pos, empty_pos, empty_val)

    srs, new_carry[0] = _resolve(*srs_args, hit_pos, hit_key, n_hits, carry[0])
    trigger_time, new_carry[1] = _resolve(*tim_args, hit_pos, hit_key, n_hits, carry[1])
    counter, new_carry[2] = _resolve(*ctr_args, hit_pos, hit_key, n_hits, carry[2])
    return srs, trigger_time, counter, new_carry


def gray2bin(arr):
    """Gray -> binary, matching vmm_pcapng_qa.gray2bin_np / Lua gray2bin32."""
    a = arr.astype(np.uint32)
    a = a ^ (a >> 16)
    a = a ^ (a >> 8)
    a = a ^ (a >> 4)
    a = a ^ (a >> 2)
    a = a ^ (a >> 1)
    return a.astype(np.uint16)


# ---------------------------------------------------------------------------
# streaming core
# ---------------------------------------------------------------------------

def iter_chunks(path, data_format="SRS", src_ips=None, max_packets=None,
                calibration=None, max_words=MAX_WORDS, meta=None,
                legacy_word_count=False):
    """Yield the raw hit columns (RAW_COLUMNS) one chunk at a time.

    This is the streaming core shared by `decode` (which concatenates) and
    `decode_to` (which writes each chunk straight to disk and drops it). Both
    therefore produce identical values by construction.

    `calibration` is accepted for signature compatibility and unused here: it
    only enters the derived columns (see `derive`).
    `meta`, if given, is a dict filled in once the generator is exhausted.

    Every 6-byte word after the 16-byte SRS header is decoded, as vmm-sdat's
    ParserSRS does. `legacy_word_count=True` reproduces the earlier Python
    parsers (vmm_pcapng_qa.py, vmm_decode.py, p2-decode 0.1), which read
    range(0, len - 22, 6) and so skipped the last word of every packet --
    1 word in 1492 for a full 8968-byte frame. Use it only to compare with
    stores made by those parsers.
    """
    if data_format not in ("SRS", "TRG"):
        raise ValueError(f"data_format must be 'SRS' or 'TRG', got {data_format!r}")
    if meta is None:
        meta = {}

    u8, pstart, plen, src, n_packets, t_ns = read_packets(
        path, max_packets=max_packets, times=True)

    if src_ips is None:
        found = detect_fec_ips(path)
        if not found:
            raise ValueError(
                f"{path}: no VM3 packets in the first {PROBE_PACKETS} packets."
            )
        src_ips = set(found)
    want = np.array([ip_u32(s) for s in sorted(src_ips)], dtype=np.uint32)

    # --- select VM3 packets from the wanted FEC(s) ---
    min_len = 23 if legacy_word_count else 22    # header + at least one word
    m = (plen >= min_len) & np.isin(src, want)
    pstart, plen, t_ns = pstart[m], plen[m], t_ns[m]
    if len(pstart):
        magic = (u8[pstart + 4].astype(np.uint32) << 16 |
                 u8[pstart + 5].astype(np.uint32) << 8 |
                 u8[pstart + 6].astype(np.uint32))
        m2 = magic == _VM3
        pstart, plen, t_ns = pstart[m2], plen[m2], t_ns[m2]

    if len(pstart) == 0:
        meta.update(_empty_meta(n_packets, src_ips, data_format))
        return

    # --- per-packet header: frame counter and FEC id ---
    frame_counter = (u8[pstart].astype(np.uint32) << 24 |
                     u8[pstart + 1].astype(np.uint32) << 16 |
                     u8[pstart + 2].astype(np.uint32) << 8 |
                     u8[pstart + 3].astype(np.uint32))
    # reference: (u32_at_byte4 >> 4) & 0x0F, i.e. the upper nibble of byte 7
    fec_id = ((u8[pstart + 7].astype(np.uint32) >> 4) & 0x0F).astype(np.uint8)
    capture = _capture_meta(fec_id, plen, t_ns, frame_counter)

    # --- flatten every 6-byte data word, in file order, from byte 16 ---
    if legacy_word_count:
        # for i in range(0, len(block) - 22, 6): read 6 bytes at i + 16
        span = plen - 22
        nwords = np.where(span > 0, (span + 5) // 6, 0)
    else:
        # a trailing partial word (never seen in DAQ data) is ignored
        nwords = (plen - 16) // 6
    total = int(nwords.sum())
    if total == 0:
        meta.update(_empty_meta(n_packets, src_ips, data_format))
        return

    # Split the packet list so no chunk decodes more than max_words words at
    # once. A 2 GB capture holds ~3e8 words; materialising those in one go
    # would blow the 80 % RAM kill the watcher enforces. Chunks always break on
    # packet boundaries, and marker state is carried across them.
    cum = np.cumsum(nwords)
    bounds = [0]
    while bounds[-1] < len(pstart):
        done = int(cum[bounds[-1] - 1]) if bounds[-1] > 0 else 0
        nxt = int(np.searchsorted(cum, done + max_words, side="left")) + 1
        bounds.append(min(max(nxt, bounds[-1] + 1), len(pstart)))

    carry = np.zeros((3, _NKEYS), dtype=np.uint64)
    hits_per_fec = np.zeros(16, dtype=np.int64)
    markers_per_fec = np.zeros(16, dtype=np.int64)
    n_vm3 = 0
    n_markers = 0
    n_hits = 0
    n_invalid = 0
    off_seen = set()

    for ci in range(len(bounds) - 1):
        lo, hi = bounds[ci], bounds[ci + 1]
        c_start, c_len = pstart[lo:hi], nwords[lo:hi]
        c_total = int(c_len.sum())
        if c_total == 0:
            continue

        pkt_of_word = np.repeat(np.arange(hi - lo, dtype=np.int64), c_len)
        starts = np.cumsum(c_len) - c_len

        # Copy each packet's word region into one contiguous buffer with
        # sequential slice reads. Fancy-indexing the memmap instead costs
        # ~3.5 s/GB, because every gather round-trips memmap.__getitem__.
        wbuf = np.empty(c_total * 6, dtype=np.uint8)
        for i in range(hi - lo):
            n = int(c_len[i])
            if n:
                s = int(c_start[i]) + 16
                o = int(starts[i]) * 6
                wbuf[o:o + n * 6] = u8[s:s + n * 6]
        w6 = wbuf.reshape(c_total, 6)
        d1 = w6[:, :4].copy().view(">u4").ravel().astype(np.uint32)
        d2 = w6[:, 4:6].copy().view(">u2").ravel().astype(np.uint32)
        del wbuf, w6

        is_hit = (d2 & 0x8000) != 0
        hit_pos = np.flatnonzero(is_hit)
        mrk_pos = np.flatnonzero(~is_hit)
        n_markers += len(mrk_pos)

        c_fec = fec_id[lo:hi]
        markers_per_fec += np.bincount(c_fec[pkt_of_word[mrk_pos]], minlength=16)
        h1 = d1[hit_pos]
        h2 = d2[hit_pos]
        hit_fec = c_fec[pkt_of_word[hit_pos]]
        vmm = ((h1 >> 22) & 0x1F).astype(np.uint8)
        hit_key = hit_fec.astype(np.int64) * 32 + vmm.astype(np.int64)

        srs_ts, trg_time, trg_ctr, carry = _decode_markers(
            d1, d2, mrk_pos, c_fec, pkt_of_word, data_format,
            hit_pos, hit_key, carry)
        del d1, d2

        ch = ((h2 >> 8) & 0x3F).astype(np.uint8)
        adc = ((h1 >> 12) & 0x3FF).astype(np.uint16)
        ot = ((h2 >> 14) & 0x1).astype(np.uint8)
        raw_off = ((h1 >> 27) & 0x1F).astype(np.int16)
        bcid_raw = (h1 & 0xFFF).astype(np.uint16)
        tdc = (h2 & 0xFF).astype(np.uint8)
        time_col = frame_counter[lo:hi][pkt_of_word[hit_pos]]
        offset = np.where(raw_off < 16, raw_off, raw_off - 32).astype(np.int8)

        n_hits += len(hit_pos)
        hits_per_fec += np.bincount(hit_fec, minlength=16)
        n_invalid += int((offset < np.int8(-1)).sum())
        if len(offset):
            off_seen.update(np.unique(offset).tolist())
        # packets yielding >=1 hit; pkt_of_word is non-decreasing and chunks
        # never split a packet, so distinct count == transitions + 1
        if len(hit_pos):
            pw = pkt_of_word[hit_pos]
            n_vm3 += int(np.count_nonzero(np.diff(pw))) + 1

        yield {
            "fec": hit_fec,
            "vmm": vmm,
            "time": time_col,
            "ch": ch,
            "adc": adc,
            "over_threshold": ot.astype(bool),
            "offset": offset,
            "bcid": gray2bin(bcid_raw),
            "tdc": tdc,
            "srs_timestamp": srs_ts,
            "trigger_time": trg_time,
            "trigger_counter": trg_ctr.astype(np.uint16),
        }

    meta.update({
        "n_packets": int(n_packets),
        # counts packets yielding >= 1 hit, matching the reference's
        # `if len(fec_buf) > n_before` test (marker-only packets do not count)
        "n_vm3_packets": int(n_vm3),
        "n_hits": int(n_hits),
        "fec_ips": sorted(src_ips),
        "data_format": data_format,
        "n_words": int(total),
        "n_markers": int(n_markers),
        "n_chunks": len(bounds) - 1,
        "n_invalid_offset": int(n_invalid),
        "offset_constant": bool(len(off_seen) == 1),
    })
    for fec, d in capture["per_fec"].items():
        d["hits"] = int(hits_per_fec[int(fec)])
        d["markers"] = int(markers_per_fec[int(fec)])
    meta["capture"] = capture
    if legacy_word_count:
        meta["legacy_word_count"] = True


def _capture_meta(fec_id, plen, t_ns, frame_counter):
    """Live time and data volume of the selected VM3 packets.

    Times are pcapng capture times (ns since the Unix epoch), so live_time_s is
    the wall-clock span from the first to the last VM3 packet. frame_counter
    is reported as usable only if it changes: the ALINX firmware sends one
    constant value, so lost frames cannot be counted from it.
    """
    ok = np.isfinite(t_ns)
    t0 = float(t_ns[ok].min()) if ok.any() else None
    t1 = float(t_ns[ok].max()) if ok.any() else None
    per_fec = {}
    for f in np.unique(fec_id):
        sel = fec_id == f
        per_fec[str(int(f))] = {"packets": int(sel.sum()),
                                "udp_payload_bytes": int(plen[sel].sum())}
    return {
        "t_first_ns": t0,
        "t_last_ns": t1,
        "live_time_s": (t1 - t0) / 1e9 if t0 is not None else None,
        "udp_payload_bytes": int(plen.sum()),
        "frame_counter_changes": bool(len(frame_counter) > 1 and
                                      (frame_counter != frame_counter[0]).any()),
        "per_fec": per_fec,
    }


def _empty_meta(n_packets, src_ips, data_format):
    return {
        "n_packets": int(n_packets), "n_vm3_packets": 0, "n_hits": 0,
        "fec_ips": sorted(src_ips), "data_format": data_format,
        "n_words": 0, "n_markers": 0, "n_chunks": 0,
        "n_invalid_offset": 0, "offset_constant": False,
    }
