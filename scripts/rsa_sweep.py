"""Budget-matched test-time-scaling sweep: does RSA latent population search
beat plain breadth on EqR, at EQUAL compute?

This is THE core experiment for the evolution-x-EqR direction (idea ③). It holds
the per-puzzle forward budget fixed (pop x rounds x depth latent-recursion steps)
and compares three search strategies, decoding all of them the SAME way (so the
only thing that varies is the *search*, not the *selection*):

  - breadth      : pop independent candidates, no recombination (the baseline)
  - rsa_conv     : each round, recombine in latent space weighted by convergence
                   residual (settle deeper into attractors)
  - rsa_consensus: each round, recombine weighted by agreement with the per-cell
                   majority board (pull the population toward the agreed solution)

For each strategy we report, at every budget point:
  - top1      : answer = most-converged candidate         (repo's selection)
  - majority  : per-cell vote over the top-k by convergence
  - any (pass@pop) : did ANY candidate solve it -> the CAPABILITY CEILING

The headline figure is `top1`/`majority` vs compute, with `any` as the ceiling.
If rsa_* sits above breadth at equal budget, latent recombination helps. If not,
that's an honest negative result -> pivot. Either way you get a paper figure.

Inference only; runs on A100 (CUDA), or M5 (MPS/CPU) at smaller settings. Reuses
the solver primitives from rsa_search.py.

Usage (from repo root, after downloading a checkpoint + data):
  scripts/download_artifacts.sh --with-ckpts            # data + checkpoints
  python3 scripts/rsa_sweep.py \
      --checkpoint downloaded_checkpoints/sudoku-extreme/eqr.pth \
      --data-dir data/sudoku-extreme --split test \
      --num-puzzles 512 --out runs/rsa_sweep.json

A day of A100: bump --num-puzzles to the full test set and add budget points.
Quick laptop check: --num-puzzles 32 --points 16x3x6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.rsa_search import (  # noqa: E402
    IGNORE_LABEL_ID,
    build_model,
    decode,
    deepen,
    diverse_init,
    exact_match,
    load_arch_config,
    load_checkpoint,
    load_puzzles,
    majority_vote,
    pick_device,
    read_metadata,
)


# --------------------------------------------------------------------------- #
def recombine_by_fitness(z_H, z_L, fitness, n_children, n_parents, temp, mutation):
    """Latent crossover where HIGHER fitness => more likely / heavier parent."""
    pop = z_H.shape[0]
    w = torch.softmax(fitness / max(temp, 1e-6), dim=0)
    kids_H, kids_L = [], []
    for _ in range(n_children):
        idx = torch.multinomial(w, num_samples=min(n_parents, pop), replacement=False)
        pw = torch.softmax(fitness[idx] / max(temp, 1e-6), dim=0).view(-1, 1, 1).to(z_H.dtype)
        kids_H.append((z_H[idx] * pw).sum(0, keepdim=True))
        kids_L.append((z_L[idx] * pw).sum(0, keepdim=True))
        if mutation > 0:
            kids_H[-1] = kids_H[-1] + torch.randn_like(kids_H[-1]) * mutation
            kids_L[-1] = kids_L[-1] + torch.randn_like(kids_L[-1]) * mutation
    return torch.cat(kids_H, 0), torch.cat(kids_L, 0)


@torch.inference_mode()
def solve_one(model, inp, pid, label, method, pop, rounds, depth, args, device, dtype) -> Dict[str, bool]:
    """Solve one puzzle with `method`, fixed budget = pop*rounds*depth deepen steps.
    Selection is identical across methods (top1/majority by convergence)."""
    seq_len = model.config.seq_len
    x = model.inner._input_embeddings(inp.view(1, -1).repeat(pop, 1).to(device), pid.view(1).repeat(pop).to(device))
    cos_sin = model.inner._cos_sin()
    label = label.to(device)

    z_H, z_L = diverse_init(model, pop, seq_len, device, dtype)
    total_per_candidate = rounds * depth

    if method == "breadth":
        z_H, z_L, resid = deepen(model, z_H, z_L, x, cos_sin, total_per_candidate)
        preds = decode(model, z_H)
    else:
        resid = None
        for r in range(rounds):
            z_H, z_L, resid = deepen(model, z_H, z_L, x, cos_sin, depth)
            preds = decode(model, z_H)
            if r == rounds - 1:
                break
            if method == "rsa_conv":
                fitness = -resid  # lower residual = fitter
            elif method == "rsa_consensus":
                C = majority_vote(preds)                       # agreed board
                fitness = (preds == C).float().mean(dim=1)     # agreement fraction
            else:
                raise ValueError(method)
            order = torch.argsort(fitness, descending=True)
            elite = order[: args.topk_elite]
            kH, kL = recombine_by_fitness(z_H, z_L, fitness, pop - args.topk_elite, args.parents, args.temp, args.mutation)
            z_H = torch.cat([z_H[elite], kH], 0)
            z_L = torch.cat([z_L[elite], kL], 0)

    # ---- identical selection for every method: by convergence residual ----
    conv_order = torch.argsort(resid)               # lower = more converged
    top1 = preds[conv_order[0]]
    maj = majority_vote(preds[conv_order[: args.topk_elite]])
    mask = label != IGNORE_LABEL_ID
    any_correct = bool((((preds == label) | ~mask).all(dim=1)).any().item())
    return {
        "top1": exact_match(top1, label),
        "majority": exact_match(maj, label),
        "any": any_correct,
    }


def parse_points(spec: str) -> List[Tuple[int, int, int]]:
    """'16x3x6,32x4x8' -> [(16,3,6),(32,4,8)] as (pop, rounds, depth)."""
    out = []
    for chunk in spec.split(","):
        p, r, d = (int(v) for v in chunk.lower().split("x"))
        out.append((p, r, d))
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch-config", default="config/arch/eqr.yaml")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-puzzles", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--methods", default="breadth,rsa_conv,rsa_consensus")
    # budget ladder: pop x rounds x depth, increasing total compute
    ap.add_argument("--points", default="16x3x6,32x4x8,48x5x10,64x6x12")
    ap.add_argument("--topk-elite", type=int, default=8)
    ap.add_argument("--parents", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--mutation", type=float, default=0.02)
    ap.add_argument("--out", default="runs/rsa_sweep.json")
    ap.add_argument("--seed", type=int, default=0, help="run with several seeds for +/-std")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    methods = args.methods.split(",")
    points = parse_points(args.points)
    max_pop = max(p for p, _, _ in points)

    print(f"[info] device={device} dtype={args.dtype} methods={methods}")
    print(f"[info] budget points (pop x rounds x depth): {points}")

    meta = read_metadata(args.data_dir, args.split)
    puzzles = load_puzzles(args.data_dir, args.split, meta, args.num_puzzles)
    model = build_model(load_arch_config(args.arch_config), meta, max_pop, device, dtype)
    n_loaded, n_missing = load_checkpoint(model, args.checkpoint)
    print(f"[info] loaded {n_loaded} tensors ({n_missing} missing); {len(puzzles)} puzzles\n")

    # results[(pop,rounds,depth)][method][metric] = accuracy
    results: Dict[str, Dict] = {}
    for (pop, rounds, depth) in points:
        budget = pop * rounds * depth
        key = f"{pop}x{rounds}x{depth}"
        results[key] = {"budget": budget, "methods": {}}
        for method in methods:
            tally = {"top1": 0, "majority": 0, "any": 0}
            t0 = time.time()
            for (inp, pid, lab) in puzzles:
                res = solve_one(model, inp, pid, lab, method, pop, rounds, depth, args, device, dtype)
                for k in tally:
                    tally[k] += int(res[k])
            n = len(puzzles)
            acc = {k: tally[k] / n for k in tally}
            results[key]["methods"][method] = acc
            dt = time.time() - t0
            print(f"  [{key} budget={budget:5d}] {method:14s} "
                  f"top1={acc['top1']:.3f} maj={acc['majority']:.3f} any={acc['any']:.3f}  ({dt:.0f}s)")
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"[info] saved -> {args.out}")

    # ---- summary table: top1 accuracy, method x budget ----
    print("\n=== top1 exact-accuracy (rows=method, cols=budget) ===")
    header = "method".ljust(15) + "".join(f"{results[k]['budget']:>8d}" for k in results)
    print(header)
    for method in methods:
        row = method.ljust(15) + "".join(f"{results[k]['methods'][method]['top1']:>8.3f}" for k in results)
        print(row)
    print("ceiling(any) ".ljust(15) + "".join(
        f"{max(results[k]['methods'][m]['any'] for m in methods):>8.3f}" for k in results))

    # ---- optional plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        budgets = [results[k]["budget"] for k in results]
        plt.figure(figsize=(6, 4))
        for method in methods:
            plt.plot(budgets, [results[k]["methods"][method]["top1"] for k in results], marker="o", label=method)
        plt.plot(budgets, [max(results[k]["methods"][m]["any"] for m in methods) for k in results],
                 "k--", alpha=0.5, label="ceiling (any)")
        plt.xlabel("compute budget (pop x rounds x depth)")
        plt.ylabel("exact accuracy (top1)")
        plt.title("RSA vs breadth: test-time scaling")
        plt.legend()
        plt.tight_layout()
        png = os.path.splitext(args.out)[0] + ".png"
        plt.savefig(png, dpi=130)
        print(f"[info] plot -> {png}")
    except ImportError:
        print("[note] matplotlib not installed; skipped plot")


if __name__ == "__main__":
    main()
