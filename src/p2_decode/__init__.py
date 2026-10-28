"""p2-decode: SRS/VMM3a raw data to hits, in electronics coordinates only.

    from p2_decode import decode
    hits, meta = decode("capture.pcapng")

Layers, bottom up:
    pcapng       pcapng container and Ethernet/IPv4/UDP, vectorised over packets
    srs          SRS frames and 6-byte VMM words; marker state machine
    calibration  derived columns (timing, calibrated ADC); calibration loader
    api, store   whole-capture decode in memory, or streamed to a column store
"""

__version__ = "0.1.0"

from .api import decode                                         # noqa: E402
from .calibration import derive, load_calibration               # noqa: E402
from .pcapng import read_packets                                # noqa: E402
from .schema import (CLOCK_PERIOD_NS, COLUMN_DTYPES, COLUMNS,   # noqa: E402
                     DERIVED_COLUMNS, RAW_COLUMNS, SCHEMA_VERSION, empty_hits)
from .srs import MAX_WORDS, detect_fec_ips, iter_chunks         # noqa: E402
from .store import HitWriter, decode_to, load_frame, load_hits  # noqa: E402

__all__ = [
    "decode", "decode_to", "load_hits", "load_frame", "derive",
    "load_calibration", "iter_chunks", "read_packets", "detect_fec_ips",
    "HitWriter", "empty_hits", "COLUMNS", "RAW_COLUMNS", "DERIVED_COLUMNS",
    "COLUMN_DTYPES", "CLOCK_PERIOD_NS", "MAX_WORDS", "SCHEMA_VERSION",
    "__version__",
]
