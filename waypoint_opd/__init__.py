"""Waypoint-OPD: gate an RL-induced policy shift by whether its halves agree."""

from .gate import agreement, centre, gate, gated_shift, shifts
from .loss import endpoint_opd_loss, gather_logps, opd_loss, topk_candidates

__version__ = "0.1.0"
__all__ = [
    "agreement",
    "centre",
    "endpoint_opd_loss",
    "gate",
    "gated_shift",
    "gather_logps",
    "opd_loss",
    "shifts",
    "topk_candidates",
]
