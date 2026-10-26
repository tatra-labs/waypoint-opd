"""A post-training run where we know which shifts were real.

The gate's premise is that a prefix where the first half of an RL run and the second half disagree
is a prefix where the run changed its mind, and that the residue left at the endpoint there is an
artefact rather than a preference worth copying. On real checkpoints that premise cannot be checked,
because nobody knows which of a teacher's endpoint shifts were durable.

So build a run where it is known. Every state gets one of two histories:

* **steady** — both halves push the same way. The endpoint shift is durable, and a student should
  take all of it.
* **wandering** — the first half pushes hard one way, the second half pushes back less hard. A
  non-zero endpoint shift survives, in a direction that meant nothing. A student should take none
  of it.

Then distil into a *different* student three ways — no transfer, endpoint-only, gated — and measure
how far each lands from the student it should have become: its own initial policy plus the durable
shifts and nothing else.

    python experiments/synthetic.py                 # the headline comparison
    python experiments/synthetic.py --sweep wander  # over the share of wandering states
    python experiments/synthetic.py --sweep honest  # over how often disagreement really is noise
    python experiments/synthetic.py --sweep k       # over how many candidates are scored
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waypoint_opd import gated_shift, opd_loss  # noqa: E402

ARMS = ("none", "endpoint", "waypoint")


def make_run(states, vocab, wander, honest, generator):
    """One synthetic post-training run. Returns the three teachers and the truth about them."""

    def directions():
        raw = torch.randn(states, vocab, generator=generator)
        return raw - raw.mean(dim=-1, keepdim=True)

    initial = torch.randn(states, vocab, generator=generator)
    durable_direction = directions()
    spurious_direction = directions()

    wandering = torch.rand(states, generator=generator) < wander
    # Among the wandering states, the ones whose residue really is noise. The rest are states
    # where the run changed its mind and the endpoint shift was worth having after all, which is
    # exactly the case the gate gets wrong.
    noise = wandering & (torch.rand(states, generator=generator) < honest)

    def scale(low, high):
        return (low + (high - low) * torch.rand(states, generator=generator)).unsqueeze(-1)

    forward_a, forward_b = scale(0.5, 1.5), scale(0.5, 1.5)
    out, back = scale(1.0, 2.0), scale(0.3, 0.9)

    steady_early = forward_a * durable_direction
    steady_late = forward_b * durable_direction
    wander_early = out * spurious_direction
    wander_late = -back * out * spurious_direction

    is_wandering = wandering.unsqueeze(-1)
    early = torch.where(is_wandering, wander_early, steady_early)
    late = torch.where(is_wandering, wander_late, steady_late)

    waypoint = initial + early
    final = waypoint + late
    # What a student should end up having absorbed: the endpoint shift everywhere except where it
    # is pure residue.
    durable = torch.where(noise.unsqueeze(-1), torch.zeros_like(early), early + late)
    return {
        "initial": initial,
        "waypoint": waypoint,
        "final": final,
        "durable": durable,
        "wandering": wandering,
        "noise": noise,
    }


def distil(run, arm, states, vocab, k, steps, beta, learning_rate, generator, device):
    """Train a student with one transfer rule and report how close it got to the ideal."""
    reference = torch.randn(states, vocab, generator=generator).to(device)
    ideal = reference + run["durable"].to(device)

    teachers = {
        name: torch.log_softmax(run[name].to(device), dim=-1)
        for name in ("initial", "waypoint", "final")
    }
    student = reference.clone().requires_grad_(arm != "none")
    if arm == "none":
        return _report(student, ideal, run, None)

    optimiser = torch.optim.Adam([student], lr=learning_rate)
    gates = None
    for _ in range(steps):
        # On-policy: the candidates are whatever the student currently prefers.
        ids = student.detach().topk(k, dim=-1).indices
        loss, parts = opd_loss(
            student.gather(-1, ids),
            teachers["initial"].gather(-1, ids),
            teachers["waypoint"].gather(-1, ids),
            teachers["final"].gather(-1, ids),
            logp_reference=torch.log_softmax(reference, dim=-1).gather(-1, ids),
            beta=beta,
            gated=arm == "waypoint",
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        gates = parts["gate"].detach()
    return _report(student.detach(), ideal, run, gates)


def _report(student, ideal, run, gates):
    target = torch.log_softmax(ideal, dim=-1)
    divergence = (target.exp() * (target - torch.log_softmax(student, dim=-1))).sum(-1)
    wandering = run["wandering"].to(student.device)
    result = {
        "kl": divergence.mean().item(),
        "kl_steady": divergence[~wandering].mean().item(),
        "kl_wandering": divergence[wandering].mean().item() if wandering.any() else 0.0,
    }
    if gates is not None:
        result["gate_steady"] = gates[~wandering].mean().item()
        result["gate_wandering"] = (
            gates[wandering].mean().item() if wandering.any() else float("nan")
        )
    return result


def one_setting(args, wander, honest, seed):
    generator = torch.Generator().manual_seed(seed)
    run = make_run(args.states, args.vocab, wander, honest, generator)
    out = {}
    for arm in ARMS:
        arm_generator = torch.Generator().manual_seed(seed + 1)
        out[arm] = distil(
            run,
            arm,
            args.states,
            args.vocab,
            args.k,
            args.steps,
            args.beta,
            args.lr,
            arm_generator,
            args.device,
        )
    out["wandering_share"] = run["wandering"].float().mean().item()
    return out


def average(results, arm, field):
    values = [result[arm][field] for result in results if field in result[arm]]
    return sum(values) / len(values) if values else float("nan")


def headline(args):
    results = [
        one_setting(args, args.wander, args.honest, args.seed + index)
        for index in range(args.seeds)
    ]
    print(
        "%d states, %d candidates of %d, %.0f%% wandering, %d seeds\n"
        % (args.states, args.k, args.vocab, 100 * args.wander, args.seeds)
    )
    print("%-10s %10s %10s %10s" % ("transfer", "KL", "KL steady", "KL wander"))
    for arm in ARMS:
        print(
            "%-10s %10.4f %10.4f %10.4f"
            % (
                arm,
                average(results, arm, "kl"),
                average(results, arm, "kl_steady"),
                average(results, arm, "kl_wandering"),
            )
        )
    print(
        "\ngate on steady states    %.3f\ngate on wandering states %.3f"
        % (
            average(results, "waypoint", "gate_steady"),
            average(results, "waypoint", "gate_wandering"),
        )
    )
    endpoint, waypoint = average(results, "endpoint", "kl"), average(results, "waypoint", "kl")
    print(
        "\nwaypoint is %.1f%% closer to the ideal student than endpoint-only"
        % (100.0 * (endpoint - waypoint) / endpoint)
    )
    return results


def sweep(args, over):
    if over == "k":
        xs = [2, 4, 8, 16, 32, args.vocab]
    else:
        xs = [index / 5.0 for index in range(0, 6)]
    rows = []
    for value in xs:
        wander = value if over == "wander" else args.wander
        honest = value if over == "honest" else args.honest
        if over == "k":
            args.k = value
        results = [
            one_setting(args, wander, honest, args.seed + index) for index in range(args.seeds)
        ]
        row = {"x": value}
        for arm in ARMS:
            row[arm] = average(results, arm, "kl")
            row[arm + "_wandering"] = average(results, arm, "kl_wandering")
        rows.append(row)
        print(
            "%s=%.1f   none %.4f   endpoint %.4f   waypoint %.4f"
            % (over, value, row["none"], row["endpoint"], row["waypoint"]),
            flush=True,
        )
    return rows


def plot(rows, over, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "wander": "share of states where the run changed its mind",
        "honest": "how often a change of mind really was noise",
        "k": "candidates scored per state, of 64",
    }
    colours = {"none": "#999999", "endpoint": "#c2492f", "waypoint": "#2f6fc2"}
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 3.9), dpi=160)
    xs = [row["x"] for row in rows]
    for axes, suffix, title in (
        (left, "", "every state"),
        (right, "_wandering", "only the states that changed their mind"),
    ):
        for arm in ARMS:
            axes.plot(
                xs,
                [row[arm + suffix] for row in rows],
                marker="o",
                ms=3.5,
                color=colours[arm],
                label=arm,
            )
        if over == "k":
            axes.set_xscale("log", base=2)
        axes.set_xlabel(labels[over])
        axes.set_title(title, fontsize=10)
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(alpha=0.25, linewidth=0.5)
    left.set_ylabel("KL to the ideal student  (lower is better)")
    left.legend(frameon=False)
    figure.suptitle("Gating a transferred policy shift", fontsize=11)
    figure.tight_layout()
    figure.savefig(path)
    print("wrote %s" % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--states", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--k", type=int, default=8, help="candidates per state")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--wander", type=float, default=0.3)
    parser.add_argument(
        "--honest",
        type=float,
        default=1.0,
        help="probability that a change of mind really was noise",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sweep", choices=("wander", "honest", "k"))
    parser.add_argument("--out", help="write the numbers here as JSON")
    parser.add_argument("--figure", help="write the sweep plot here")
    args = parser.parse_args()

    if args.sweep:
        rows = sweep(args, args.sweep)
        if args.figure:
            plot(rows, args.sweep, args.figure)
        payload = {"sweep": args.sweep, "rows": rows}
    else:
        payload = {"headline": headline(args)}
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
