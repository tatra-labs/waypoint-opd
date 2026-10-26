import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waypoint_opd import (  # noqa: E402
    agreement,
    centre,
    endpoint_opd_loss,
    gate,
    gated_shift,
    gather_logps,
    opd_loss,
    shifts,
    topk_candidates,
)


def logp(values):
    return torch.log_softmax(torch.tensor(values, dtype=torch.float32), dim=-1)


# --- the shifts -----------------------------------------------------------------------------


def test_centring_removes_the_mean():
    values = torch.tensor([[1.0, 2.0, 6.0]])
    assert torch.allclose(centre(values).mean(-1), torch.zeros(1), atol=1e-6)


def test_the_endpoint_shift_is_the_two_halves_added():
    initial, waypoint, final = (torch.randn(4, 7) for _ in range(3))
    early, late, endpoint = shifts(initial, waypoint, final)
    assert torch.allclose(endpoint, early + late, atol=1e-5)


def test_a_uniform_change_of_confidence_is_not_a_shift():
    """Relative transfer must ignore a teacher that simply got more or less certain overall."""
    initial = torch.randn(3, 5)
    final = torch.randn(3, 5)
    plain = centre(final - initial)
    louder = centre((final + 4.0) - initial)
    assert torch.allclose(plain, louder, atol=1e-5)


# --- the gate -------------------------------------------------------------------------------


def test_halves_that_agree_gate_to_one():
    direction = centre(torch.randn(6, 9))
    initial = torch.randn(6, 9)
    waypoint = initial + 0.7 * direction
    final = waypoint + 1.3 * direction
    parts = gated_shift(initial, waypoint, final)
    assert torch.allclose(parts["gate"], torch.ones(6), atol=1e-4)
    assert torch.allclose(parts["signal"], parts["shift"], atol=1e-5)


def test_halves_that_oppose_gate_to_zero():
    direction = centre(torch.randn(6, 9))
    initial = torch.randn(6, 9)
    waypoint = initial + 1.5 * direction
    final = waypoint - 0.6 * direction
    parts = gated_shift(initial, waypoint, final)
    assert torch.allclose(parts["cos"], -torch.ones(6), atol=1e-4)
    assert torch.allclose(parts["gate"], torch.zeros(6), atol=1e-6)
    # ... even though a perfectly real-looking endpoint shift survived.
    assert parts["shift"].norm(dim=-1).min() > 0.1
    assert torch.allclose(parts["signal"], torch.zeros_like(parts["signal"]), atol=1e-6)


def test_a_right_angle_gates_to_nothing_too():
    initial = torch.zeros(1, 4)
    waypoint = initial + torch.tensor([[1.0, -1.0, 0.0, 0.0]])
    final = waypoint + torch.tensor([[0.0, 0.0, 1.0, -1.0]])
    parts = gated_shift(initial, waypoint, final)
    assert abs(float(parts["cos"])) < 1e-5
    assert float(parts["gate"]) < 1e-5


def test_a_flat_half_falls_back_rather_than_guessing():
    initial = torch.zeros(1, 4)
    waypoint = initial.clone()  # the first half changed nothing: the cosine is meaningless
    final = waypoint + torch.tensor([[1.0, -1.0, 0.5, -0.5]])
    assert float(gated_shift(initial, waypoint, final, fallback=1.0)["gate"]) == 1.0
    assert float(gated_shift(initial, waypoint, final, fallback=0.0)["gate"]) == 0.0
    parts = gated_shift(initial, waypoint, final, fallback=1.0)
    assert not bool(parts["defined"].any())
    assert torch.allclose(parts["signal"], parts["shift"], atol=1e-6)


def test_the_gate_clamps_rather_than_flipping():
    cos = torch.tensor([-1.0, -0.2, 0.0, 0.4, 1.0])
    defined = torch.ones(5, dtype=torch.bool)
    assert torch.allclose(gate(cos, defined), torch.tensor([0.0, 0.0, 0.0, 0.4, 1.0]))


def test_agreement_reports_where_it_is_meaningful():
    early = torch.tensor([[1.0, -1.0], [0.0, 0.0]])
    late = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    cos, defined = agreement(early, late)
    assert defined.tolist() == [True, False]
    assert pytest.approx(float(cos[0]), abs=1e-5) == 1.0
    assert float(cos[1]) == 0.0


# --- the loss -------------------------------------------------------------------------------


def test_the_loss_is_zero_when_the_student_is_already_there():
    reference = torch.log_softmax(torch.randn(5, 6), dim=-1)
    direction = centre(torch.randn(5, 6))
    initial = torch.randn(5, 6)
    waypoint = initial + direction
    final = waypoint + direction
    signal = centre(final - initial)
    student = reference + signal
    loss, _ = opd_loss(
        student,
        initial,
        waypoint,
        final,
        logp_reference=reference,
        beta=1.0,
    )
    assert float(loss) < 1e-6


def test_the_loss_moves_the_student_towards_the_signal():
    torch.manual_seed(0)
    reference = torch.log_softmax(torch.randn(8, 10), dim=-1)
    direction = centre(torch.randn(8, 10))
    initial = torch.randn(8, 10)
    waypoint = initial + direction
    final = waypoint + direction
    student = reference.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([student], lr=0.1)
    first = None
    for _ in range(200):
        loss, _ = opd_loss(student, initial, waypoint, final, logp_reference=reference)
        first = float(loss.detach()) if first is None else first
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert float(loss) < first / 50


def test_gating_removes_exactly_the_positions_that_changed_their_mind():
    direction = centre(torch.randn(2, 6))
    initial = torch.randn(2, 6)
    agreeing = initial + direction, initial + 2.0 * direction
    opposing = initial + 2.0 * direction, initial + 0.8 * direction
    reference = torch.log_softmax(torch.randn(2, 6), dim=-1)

    _, agree_parts = opd_loss(reference, initial, *agreeing, logp_reference=reference)
    _, oppose_parts = opd_loss(reference, initial, *opposing, logp_reference=reference)
    # An agreeing position has something to learn; an opposing one is left alone entirely.
    assert float(agree_parts["per_position"].mean()) > 1e-3
    assert float(oppose_parts["per_position"].mean()) < 1e-9
    # Endpoint-only transfer cannot tell them apart: both look like work to do.
    _, endpoint_parts = endpoint_opd_loss(reference, initial, opposing[1], logp_reference=reference)
    assert float(endpoint_parts["per_position"].mean()) > 1e-3


def test_endpoint_and_gated_transfer_agree_when_the_halves_do():
    direction = centre(torch.randn(4, 7))
    initial = torch.randn(4, 7)
    waypoint = initial + direction
    final = waypoint + 1.5 * direction
    reference = torch.log_softmax(torch.randn(4, 7), dim=-1)
    gated, _ = opd_loss(reference, initial, waypoint, final, logp_reference=reference)
    plain, _ = endpoint_opd_loss(reference, initial, final, logp_reference=reference)
    assert float(gated) == pytest.approx(float(plain), rel=1e-4)


def test_a_mask_restricts_the_average():
    initial, waypoint, final = (torch.randn(4, 5) for _ in range(3))
    student = torch.randn(4, 5)
    mask = torch.tensor([True, False, False, False])
    loss, parts = opd_loss(student, initial, waypoint, final, mask=mask)
    assert float(loss) == pytest.approx(float(parts["per_position"][0]), rel=1e-5)


def test_gated_transfer_insists_on_a_waypoint():
    initial, final = torch.randn(2, 3), torch.randn(2, 3)
    with pytest.raises(ValueError):
        opd_loss(torch.randn(2, 3), initial, None, final, gated=True)
    with pytest.raises(ValueError):
        opd_loss(torch.randn(2, 3), initial, gated=False)


# --- the candidate helpers ------------------------------------------------------------------


def test_candidates_come_from_the_student():
    logits = torch.tensor([[0.0, 5.0, 1.0, 4.0]])
    ids, values = topk_candidates(logits, 2)
    assert ids.tolist() == [[1, 3]]
    assert values.tolist() == [[5.0, 4.0]]


def test_teacher_scores_are_full_vocabulary_log_probabilities():
    logits = torch.tensor([[0.0, 5.0, 1.0, 4.0]])
    ids = torch.tensor([[1, 3]])
    got = gather_logps(logits, ids)
    want = torch.log_softmax(logits, dim=-1).gather(-1, ids)
    assert torch.allclose(got, want, atol=1e-6)


# --- the synthetic run ----------------------------------------------------------------------


def test_the_synthetic_run_separates_wandering_from_steady():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    import synthetic

    generator = torch.Generator().manual_seed(3)
    run = synthetic.make_run(128, 32, wander=0.5, honest=1.0, generator=generator)
    parts = gated_shift(
        torch.log_softmax(run["initial"], dim=-1),
        torch.log_softmax(run["waypoint"], dim=-1),
        torch.log_softmax(run["final"], dim=-1),
    )
    wandering = run["wandering"]
    assert float(parts["gate"][~wandering].mean()) > 0.99
    assert float(parts["gate"][wandering].mean()) < 0.01
    # The wandering states still carry a sizeable endpoint shift, which is the whole problem.
    assert float(parts["shift"][wandering].norm(dim=-1).mean()) > 0.5
