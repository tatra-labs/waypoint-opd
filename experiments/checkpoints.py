"""Does a real post-training run actually change its mind?

The gate is only worth paying for if early and late relative preference shifts disagree often
enough on the prefixes a student actually visits. That is an empirical question about real
checkpoints, and it needs a post-training run whose *intermediate* checkpoint was released. OLMo 2
is one: `allenai/OLMo-2-0425-1B-SFT` -> `-DPO` -> `-Instruct` is one pipeline, three public
checkpoints, so Initial, Waypoint and Final are all on the shelf.

A student generates the trajectories; its own top-k candidates at every generated position are what
the three frozen teachers get scored on; and the early, late and endpoint shifts fall out of that.
Nothing is trained here — this measures the signal the gate would act on.

    python experiments/checkpoints.py --device cuda --max-new-tokens 64

The teachers are loaded one at a time and freed, so the peak cost is one 1B model plus the cached
candidate scores. It runs on 8 GB.
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waypoint_opd import gather_logps, gated_shift  # noqa: E402

CHAIN = {
    "initial": "allenai/OLMo-2-0425-1B-SFT",
    "waypoint": "allenai/OLMo-2-0425-1B-DPO",
    "final": "allenai/OLMo-2-0425-1B-Instruct",
}
DEFAULT_STUDENT = "allenai/OLMo-2-0425-1B"
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(os.path.dirname(HERE), "data", "prompts.txt")


def chain_from(args):
    """The three checkpoints to compare, overridable for any other post-training run."""
    return {
        "initial": args.initial,
        "waypoint": args.waypoint,
        "final": args.final,
    }


def load(name, device, dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(device)
    model.eval()
    return model


def release(model, device):
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def read_prompts(path, limit):
    with open(path, encoding="utf-8") as fh:
        prompts = [line.strip() for line in fh if line.strip()]
    return prompts[:limit] if limit else prompts


def apply_template(tokenizer, prompt):
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return prompt + "\n"


@torch.no_grad()
def roll_out(args, tokenizer):
    """Generate one completion per prompt with the student, then read off its candidates."""
    student = load(args.student, args.device, args.dtype)
    prompts = read_prompts(args.prompts, args.prompts_limit)
    rollouts = []
    for index, prompt in enumerate(prompts):
        text = apply_template(tokenizer, prompt)
        encoded = tokenizer(text, return_tensors="pt").to(args.device)
        generated = student.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=0.95 if args.temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        sequence = generated[0]
        start = encoded["input_ids"].shape[1]
        if sequence.shape[0] - start < 2:
            continue
        logits = student(sequence.unsqueeze(0)).logits[0].float()
        # Position p of the logits predicts token p+1, so the completion is predicted from
        # start-1 up to the last token but one.
        at = torch.arange(start - 1, sequence.shape[0] - 1, device=sequence.device)
        candidate_ids = logits[at].topk(args.k, dim=-1).indices
        rollouts.append(
            {
                "prompt": prompt,
                "sequence": sequence.cpu(),
                "at": at.cpu(),
                "candidate_ids": candidate_ids.cpu(),
                "student_logp": gather_logps(logits[at], candidate_ids).cpu(),
                "index": index,
            }
        )
        print(
            "  rolled out %2d/%d  %d positions" % (index + 1, len(prompts), at.shape[0]),
            flush=True,
        )
    release(student, args.device)
    return rollouts


@torch.no_grad()
def score(args, rollouts, name):
    """One frozen teacher's log-probabilities on every recorded candidate."""
    started = time.time()
    model = load(args.chain[name], args.device, args.dtype)
    out = []
    for rollout in rollouts:
        sequence = rollout["sequence"].to(args.device)
        logits = model(sequence.unsqueeze(0)).logits[0].float()
        at = rollout["at"].to(args.device)
        ids = rollout["candidate_ids"].to(args.device)
        out.append(gather_logps(logits[at], ids).cpu())
    release(model, args.device)
    return out, time.time() - started


def summarise(rollouts, scored, args):
    """Gate statistics over every student-visited position."""
    rows = []
    for index, rollout in enumerate(rollouts):
        parts = gated_shift(
            scored["initial"][index],
            scored["waypoint"][index],
            scored["final"][index],
            fallback=args.fallback,
        )
        rows.append(
            {
                "cos": parts["cos"],
                "gate": parts["gate"],
                "defined": parts["defined"],
                "shift_norm": parts["shift"].norm(dim=-1),
                "early_norm": parts["early"].norm(dim=-1),
                "late_norm": parts["late"].norm(dim=-1),
                "prompt": rollout["index"],
            }
        )
    joined = {
        key: torch.cat([row[key] for row in rows])
        for key in ("cos", "gate", "defined", "shift_norm", "early_norm", "late_norm")
    }
    cos, gate = joined["cos"], joined["gate"]
    defined = joined["defined"]
    norms = joined["shift_norm"]
    disagreeing = defined & (cos < 0)
    agreeing = defined & (cos > 0)
    suppressed = (norms * (1.0 - gate)).sum() / norms.sum().clamp_min(1e-9)
    return joined, {
        "positions": int(cos.numel()),
        "prompts": len(rollouts),
        "undefined_share": float((~defined).float().mean()),
        "disagreeing_share": float(disagreeing.float().mean()),
        "mean_cos": float(cos[defined].mean()) if defined.any() else float("nan"),
        "median_cos": float(cos[defined].median()) if defined.any() else float("nan"),
        "mean_gate": float(gate.mean()),
        "signal_suppressed": float(suppressed),
        "mean_shift_norm_agreeing": float(norms[agreeing].mean()) if agreeing.any() else 0.0,
        "mean_shift_norm_disagreeing": (
            float(norms[disagreeing].mean()) if disagreeing.any() else 0.0
        ),
    }


def plot(joined, stats, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    defined = joined["defined"]
    cos = joined["cos"][defined].numpy()
    norms = joined["shift_norm"][defined].numpy()

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 3.9), dpi=160)
    left.hist(cos, bins=60, color="#2f6fc2", alpha=0.85)
    left.axvline(0.0, color="#c2492f", lw=1.2)
    left.set_xlabel("cos(early shift, late shift)")
    left.set_ylabel("student-visited positions")
    left.set_title(
        "%.0f%% of positions disagree and are gated off" % (100 * stats["disagreeing_share"]),
        fontsize=10,
    )

    right.scatter(cos, norms, s=3, alpha=0.18, color="#2f6fc2", edgecolors="none")
    right.axvline(0.0, color="#c2492f", lw=1.2)
    right.set_xlabel("cos(early shift, late shift)")
    right.set_ylabel("endpoint shift magnitude")
    right.set_title(
        "the gate removes %.0f%% of the transferred signal" % (100 * stats["signal_suppressed"]),
        fontsize=10,
    )
    for axes in (left, right):
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(alpha=0.2, linewidth=0.5)
    figure.suptitle(
        "%s,  %d positions from %d prompts"
        % (stats["chain_label"], stats["positions"], stats["prompts"]),
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path)
    print("wrote %s" % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--initial", default=CHAIN["initial"], help="the checkpoint before RL")
    parser.add_argument("--waypoint", default=CHAIN["waypoint"], help="one from partway through")
    parser.add_argument("--final", default=CHAIN["final"], help="the checkpoint after")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--k", type=int, default=8, help="candidates per position")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompts", default=PROMPTS)
    parser.add_argument("--prompts-limit", type=int, default=0)
    parser.add_argument("--fallback", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/checkpoints.json")
    parser.add_argument("--figure", default="figures/checkpoints.png")
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    args.chain = chain_from(args)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.chain["final"])

    print("rolling out with %s" % args.student, flush=True)
    rollouts = roll_out(args, tokenizer)
    scored, timings = {}, {}
    for name in ("initial", "waypoint", "final"):
        print("scoring under %s" % args.chain[name], flush=True)
        scored[name], timings[name] = score(args, rollouts, name)

    joined, stats = summarise(rollouts, scored, args)
    stats["seconds_per_teacher_pass"] = timings
    stats["waypoint_overhead"] = timings["waypoint"] / (
        timings["initial"] + timings["final"]
    )
    stats["student"] = args.student
    stats["chain_label"] = " -> ".join(
        args.chain[name].split("/")[-1] for name in ("initial", "waypoint", "final")
    )
    stats["chain"] = args.chain
    stats["k"] = args.k

    print()
    for key in (
        "prompts",
        "positions",
        "mean_cos",
        "median_cos",
        "disagreeing_share",
        "undefined_share",
        "mean_gate",
        "signal_suppressed",
        "mean_shift_norm_agreeing",
        "mean_shift_norm_disagreeing",
        "waypoint_overhead",
    ):
        value = stats[key]
        print("%-30s %s" % (key, "%.4f" % value if isinstance(value, float) else value))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(stats, indent=2) + "\n")
        print("wrote %s" % args.out)
    if args.figure:
        os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)
        plot(joined, stats, args.figure)


if __name__ == "__main__":
    main()
