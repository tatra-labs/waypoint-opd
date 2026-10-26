"""The gate: does the early half of an RL run agree with the late half?

Three checkpoints of one post-training run — Initial, Waypoint, Final — are scored on the same
candidate tokens at the same prefix. Two *relative preference shifts* fall out of that:

    early = centre(logp_W - logp_I)        how preferences moved in the first half
    late  = centre(logp_F - logp_W)        how they moved in the second half

Centring is what makes them *relative*: subtracting the mean over the candidates throws away any
uniform change in confidence and keeps only the reordering. The endpoint shift

    shift = centre(logp_F - logp_I) = early + late

is what endpoint-only transfer uses, and it cannot tell "both halves pushed this way" from "the
first half pushed hard and the second half pushed back". The cosine between early and late can:

    gate = max(0, cos(early, late))

so a prefix where the run changed its mind contributes nothing, one where both halves agreed
contributes fully, and the ones in between are scaled.
"""

import torch

EPS = 1e-6


def centre(values, dim=-1):
    """Remove the mean over the candidate axis, leaving only relative preference."""
    return values - values.mean(dim=dim, keepdim=True)


def shifts(logp_initial, logp_waypoint, logp_final):
    """The early, late and endpoint relative shifts. Shapes are (..., candidates)."""
    early = centre(logp_waypoint - logp_initial)
    late = centre(logp_final - logp_waypoint)
    return early, late, early + late


def agreement(early, late, eps=EPS):
    """Cosine between the two shifts, and a mask of where it is meaningful.

    Returns `(cos, defined)`, both shaped like the leading axes. Where either shift is
    numerically flat the cosine is noise, `defined` is False, and `cos` is zero.
    """
    early_norm = early.norm(dim=-1)
    late_norm = late.norm(dim=-1)
    defined = (early_norm > eps) & (late_norm > eps)
    dot = (early * late).sum(dim=-1)
    cos = dot / (early_norm * late_norm).clamp_min(eps)
    return torch.where(defined, cos, torch.zeros_like(cos)), defined


def gate(cos, defined, fallback=1.0):
    """max(0, cos), with `fallback` wherever the cosine was not meaningful.

    The default fallback of 1.0 is the conservative one: when there is no evidence of a change
    of mind, behave exactly like endpoint-only transfer. Use 0.0 to transfer only what the
    waypoint positively vouches for.
    """
    gated = cos.clamp_min(0.0)
    return torch.where(defined, gated, torch.full_like(gated, fallback))


def gated_shift(logp_initial, logp_waypoint, logp_final, fallback=1.0, eps=EPS):
    """The supervision signal of Waypoint-OPD, and the parts it is made of.

    Returns a dict with `signal` (the gated endpoint shift, shaped like the inputs), and the
    diagnostics `shift`, `early`, `late`, `cos`, `gate` and `defined`.
    """
    early, late, endpoint = shifts(logp_initial, logp_waypoint, logp_final)
    cos, defined = agreement(early, late, eps=eps)
    weights = gate(cos, defined, fallback=fallback)
    return {
        "signal": endpoint * weights.unsqueeze(-1),
        "shift": endpoint,
        "early": early,
        "late": late,
        "cos": cos,
        "gate": weights,
        "defined": defined,
    }
