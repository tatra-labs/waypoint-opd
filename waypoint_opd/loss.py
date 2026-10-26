"""On-policy distillation of a relative policy shift, with and without the waypoint gate.

The student generates the trajectory, so the prefixes are its own. At each prefix the teachers are
asked about the student's own top candidates, and what is transferred is not the teacher's
distribution — the two models need not be the same size, and copying the teacher's absolute
preferences would fight the student's own — but how post-training *moved* the teacher's preferences
among those candidates:

    target = softmax(log pi_reference + beta * signal)
    loss   = KL(target || pi_student)

with the target detached. `pi_reference` is the student before distillation started, frozen; using
it rather than the running student is what gives the objective a fixed point, in the same way a
KL-regularised reward objective does. `signal` is the endpoint shift for endpoint-only transfer and
the gated endpoint shift for Waypoint-OPD. Everything is restricted to the candidate set and
renormalised there, which is what makes one extra frozen-teacher pass enough.
"""

import torch

from .gate import centre, gated_shift, shifts


def topk_candidates(student_logits, k):
    """The student's top-`k` next tokens at each position, and its logits on them."""
    values, ids = student_logits.topk(k, dim=-1)
    return ids, values


def gather_logps(logits, ids):
    """Full-vocabulary log-probabilities, read off at the candidate ids."""
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, ids)


def opd_loss(
    student_logits,
    logp_initial,
    logp_waypoint=None,
    logp_final=None,
    logp_reference=None,
    beta=1.0,
    gated=True,
    fallback=1.0,
    mask=None,
):
    """The loss, and the diagnostics behind it.

    `student_logits` are the student's logits on the candidate set, shaped (..., candidates), and
    the three `logp_*` arguments are the frozen teachers' log-probabilities on the same candidates.
    `logp_reference` is the frozen student reference on the same candidates; without it the running
    student is used instead, which has no fixed point and only makes sense under a fixed step
    budget. With `gated=False` the waypoint is not needed and this is endpoint-only on-policy
    distillation. `mask` is a boolean of the leading shape selecting the positions to train on.
    """
    if logp_final is None:
        raise ValueError("logp_final is required")
    student_logp = torch.log_softmax(student_logits.float(), dim=-1)

    if gated:
        if logp_waypoint is None:
            raise ValueError("gated transfer needs the waypoint checkpoint")
        parts = gated_shift(logp_initial, logp_waypoint, logp_final, fallback=fallback)
    else:
        endpoint = centre(logp_final - logp_initial)
        parts = {
            "signal": endpoint,
            "shift": endpoint,
            "gate": torch.ones(endpoint.shape[:-1], device=endpoint.device),
        }
        if logp_waypoint is not None:
            early, late, _ = shifts(logp_initial, logp_waypoint, logp_final)
            parts["early"], parts["late"] = early, late

    anchor = student_logp.detach() if logp_reference is None else logp_reference
    target = torch.softmax(anchor + beta * parts["signal"], dim=-1)
    per_position = (target * (torch.log(target.clamp_min(1e-12)) - student_logp)).sum(-1)

    if mask is None:
        loss = per_position.mean()
    else:
        weights = mask.to(per_position.dtype)
        loss = (per_position * weights).sum() / weights.sum().clamp_min(1.0)

    parts["per_position"] = per_position
    parts["loss"] = loss
    return loss, parts


def endpoint_opd_loss(
    student_logits, logp_initial, logp_final, logp_reference=None, beta=1.0, mask=None
):
    """Endpoint-only transfer: the same loss with the gate removed. The baseline."""
    return opd_loss(
        student_logits,
        logp_initial,
        None,
        logp_final,
        logp_reference=logp_reference,
        beta=beta,
        gated=False,
        mask=mask,
    )
