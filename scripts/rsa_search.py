"""RSA-style inference-time population search for EqR (M5 / MPS / CPU friendly).

This is a research prototype, not a tuned method. It adapts Recursive
Self-Aggregation (RSA, arXiv:2509.26626) and AB-MCTS-style breadth/depth search
to EqR's attractor dynamics: instead of evolving solution *strings* (as RSA does
for LLMs), we evolve a population of latent solver states (z_H, z_L). Each round
we (1) deepen every candidate by running EqR's latent recursion, (2) score each
by its convergence residual (the same signal the repo's convergence_top_k uses,
lower = more settled into an attractor), and (3) recombine: children are formed
by aggregating K parents in latent space (convergence-weighted mean + mutation
noise), with the top-E elites carried over unchanged.

We report three decoders on the same compute so you can see whether
recombination actually helps over plain different-init breadth:
  - rsa/top1        : best candidate by convergence score
  - rsa/majority    : per-cell majority vote over the elite candidates
  - breadth/majority: SAME population WITHOUT recombination (ablation baseline)

Nothing here touches flash-attn or the CUDA optimizer, so it runs on Apple
Silicon via the SDPA attention fallback in models/layers.py. Training still needs
a CUDA box; this is inference only.

Usage (from repo root):
  # Smoke test the loop with random weights — proves it runs on your M5:
  python3 scripts/rsa_search.py --smoke

  # Real run against a downloaded checkpoint + dataset split:
  python3 scripts/rsa_search.py \
      --checkpoint downloaded_checkpoints/sudoku-extreme/eqr.pth \
      --data-dir data/sudoku-extreme \
      --split test --num-puzzles 64 \
      --pop 32 --rounds 4 --depth 8 --topk-elite 8 --parents 4

Tune the population/rounds/depth down if MPS memory is tight; pop=16, rounds=3,
depth=6 is comfortable on a laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Import only the model — never pretrain.py / the CUDA optimizer.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.eqr import EqRModel  # noqa: E402
from models.common import trunc_normal_init_  # noqa: E402

IGNORE_LABEL_ID = -100


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_arch_config(path: str) -> Dict:
    """Read config/arch/eqr.yaml, keeping only EqRConfig fields."""
    import yaml

    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    drop = {"name", "short_name", "loss"}
    return {k: v for k, v in raw.items() if k not in drop}


def read_metadata(data_dir: str, split: str) -> Dict:
    with open(os.path.join(data_dir, split, "dataset.json"), "r") as f:
        return json.load(f)


def build_model(arch_cfg: Dict, meta: Dict, pop: int, device: torch.device, dtype: torch.dtype) -> EqRModel:
    cfg = dict(arch_cfg)
    cfg.update(
        batch_size=pop,
        seq_len=meta["seq_len"],
        vocab_size=meta["vocab_size"],
        num_puzzle_identifiers=meta["num_puzzle_identifiers"],
        forward_dtype={torch.float32: "float32", torch.bfloat16: "bfloat16", torch.float16: "float16"}[dtype],
    )
    model = EqRModel(cfg)
    model.eval().to(device)
    return model


def load_checkpoint(model: EqRModel, path: str) -> Tuple[int, int]:
    """Load a checkpoint, normalizing common wrapper prefixes (DDP / compile /
    loss head). Returns (#loaded, #missing) so the caller can sanity-check."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    # Normalize keys so they start at "inner."
    norm = {}
    for k, v in state.items():
        if "inner." in k:
            k = "inner." + k.split("inner.", 1)[1]
        norm[k] = v
    result = model.load_state_dict(norm, strict=False)
    n_loaded = len(model.state_dict()) - len(result.missing_keys)
    if result.missing_keys:
        # H_init/L_init are persistent=False buffers and are expected to be absent.
        real_missing = [k for k in result.missing_keys if not k.endswith(("H_init", "L_init"))]
        if real_missing:
            print(f"[warn] {len(real_missing)} missing keys, e.g. {real_missing[:4]}")
    if result.unexpected_keys:
        print(f"[warn] {len(result.unexpected_keys)} unexpected keys, e.g. {result.unexpected_keys[:4]}")
    return n_loaded, len(result.missing_keys)


# --------------------------------------------------------------------------- #
# EqR latent solver primitives
# --------------------------------------------------------------------------- #
def diverse_init(model: EqRModel, pop: int, seq_len: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """N distinct latent seeds, matching the repo's different-init eval."""
    d = model.config.hidden_size
    z_H = trunc_normal_init_(torch.empty(pop, seq_len, d, dtype=dtype, device=device), std=model.config.H_init_std)
    z_L = trunc_normal_init_(torch.empty(pop, seq_len, d, dtype=dtype, device=device), std=model.config.L_init_std)
    return z_H, z_L


@torch.inference_mode()
def deepen(model: EqRModel, z_H, z_L, x, cos_sin, steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run `steps` of EqR latent recursion. Returns refined (z_H, z_L) and a
    per-candidate convergence residual = mean latent step-norm over the run
    (lower => closer to a fixed point / attractor)."""
    seq = {"cos_sin": cos_sin}
    resid = torch.zeros(z_H.shape[0], dtype=torch.float32, device=z_H.device)
    for _ in range(steps):
        prev = z_H
        z_H, z_L = model.inner.latent_recursion(z_H, z_L, x, seq)
        resid += (z_H - prev).flatten(1).to(torch.float32).norm(dim=1)
    return z_H, z_L, resid / max(1, steps)


@torch.inference_mode()
def decode(model: EqRModel, z_H) -> torch.Tensor:
    """z_H -> predicted token ids [pop, out_len]."""
    out = z_H[:, model.config.puzzle_emb_len:]
    logits = model.inner.lm_head(out)
    return logits.argmax(dim=-1)


def recombine(z_H, z_L, scores, n_children: int, n_parents: int, temp: float, mutation: float):
    """Convergence-weighted latent crossover. Each child = weighted mean of
    `n_parents` sampled parents (better-converged => higher weight) + Gaussian
    mutation. This is the 'self-aggregation' step, done in attractor space."""
    pop = z_H.shape[0]
    w = torch.softmax(-scores / max(temp, 1e-6), dim=0)  # lower residual -> higher prob
    kids_H, kids_L = [], []
    for _ in range(n_children):
        idx = torch.multinomial(w, num_samples=min(n_parents, pop), replacement=False)
        pw = torch.softmax(-scores[idx] / max(temp, 1e-6), dim=0).view(-1, 1, 1).to(z_H.dtype)
        cH = (z_H[idx] * pw).sum(0, keepdim=True)
        cL = (z_L[idx] * pw).sum(0, keepdim=True)
        if mutation > 0:
            cH = cH + torch.randn_like(cH) * mutation
            cL = cL + torch.randn_like(cL) * mutation
        kids_H.append(cH)
        kids_L.append(cL)
    return torch.cat(kids_H, 0), torch.cat(kids_L, 0)


def majority_vote(preds: torch.Tensor) -> torch.Tensor:
    """Per-cell mode over a set of candidate boards [k, L] -> [L]."""
    out = []
    for col in preds.t():  # iterate cells
        vals, counts = torch.unique(col, return_counts=True)
        out.append(vals[counts.argmax()])
    return torch.stack(out)


def exact_match(pred: torch.Tensor, label: torch.Tensor) -> bool:
    mask = label != IGNORE_LABEL_ID
    return bool(((pred == label) | ~mask).all().item())


# --------------------------------------------------------------------------- #
# RSA search over one puzzle
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def rsa_solve(model, inputs, puzzle_id, label, args, device, dtype) -> Dict[str, bool]:
    seq_len = model.config.seq_len
    pop = args.pop

    inp = inputs.view(1, -1).repeat(pop, 1).to(device)
    pid = puzzle_id.view(1).repeat(pop).to(device)
    x = model.inner._input_embeddings(inp, pid)
    cos_sin = model.inner._cos_sin()

    # ----- RSA population (with recombination) -----
    z_H, z_L = diverse_init(model, pop, seq_len, device, dtype)
    best_pred, best_score = None, None
    for r in range(args.rounds):
        z_H, z_L, score = deepen(model, z_H, z_L, x, cos_sin, args.depth)
        preds = decode(model, z_H)
        # track global best-by-convergence
        b = int(score.argmin())
        if best_score is None or score[b] < best_score:
            best_score, best_pred = float(score[b]), preds[b].clone()
        if r == args.rounds - 1:
            break
        # elitism + recombination
        order = torch.argsort(score)
        elite = order[: args.topk_elite]
        eH, eL = z_H[elite], z_L[elite]
        n_children = pop - args.topk_elite
        kH, kL = recombine(z_H, z_L, score, n_children, args.parents, args.temp, args.mutation)
        z_H = torch.cat([eH, kH], 0)
        z_L = torch.cat([eL, kL], 0)

    # final elite set of the RSA population
    elite_idx = torch.argsort(score)[: args.topk_elite]
    rsa_majority = majority_vote(preds[elite_idx])

    # ----- Breadth baseline: same #candidates, same total depth, NO recombination -----
    bH, bL = diverse_init(model, pop, seq_len, device, dtype)
    bH, bL, bscore = deepen(model, bH, bL, x, cos_sin, args.depth * args.rounds)
    bpreds = decode(model, bH)
    breadth_majority = majority_vote(bpreds[torch.argsort(bscore)[: args.topk_elite]])

    label = label.to(device)
    return {
        "rsa/top1": exact_match(best_pred, label),
        "rsa/majority": exact_match(rsa_majority, label),
        "breadth/majority": exact_match(breadth_majority, label),
    }


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_puzzles(data_dir: str, split: str, meta: Dict, n: int):
    set_name = meta["sets"][0]
    base = os.path.join(data_dir, split, set_name)
    inputs = np.load(base + "__inputs.npy", mmap_mode="r")
    labels = np.load(base + "__labels.npy", mmap_mode="r")
    pids = np.load(base + "__puzzle_identifiers.npy", mmap_mode="r")
    pidx = np.load(base + "__puzzle_indices.npy", mmap_mode="r")
    n = min(n, len(inputs))
    out = []
    for i in range(n):
        # find this row's puzzle id via puzzle_indices
        p = int(np.searchsorted(pidx, i, side="right") - 1)
        lab = labels[i].astype(np.int64)
        lab[lab == (meta.get("ignore_label_id") if meta.get("ignore_label_id") is not None else -1)] = IGNORE_LABEL_ID
        out.append((
            torch.from_numpy(inputs[i].astype(np.int64)),
            torch.tensor(int(pids[p]) if p < len(pids) else 0, dtype=torch.long),
            torch.from_numpy(lab),
        ))
    return out


def make_smoke_puzzles(meta: Dict, n: int):
    """Random inputs/labels just to exercise the loop without a checkpoint."""
    L, V = meta["seq_len"], meta["vocab_size"]
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        out.append((
            torch.randint(0, V, (L,), generator=g),
            torch.tensor(0),
            torch.randint(0, V, (L,), generator=g),
        ))
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--arch-config", default="config/arch/eqr.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-puzzles", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    # search hyperparameters
    ap.add_argument("--pop", type=int, default=32, help="population size")
    ap.add_argument("--rounds", type=int, default=4, help="RSA recombination rounds")
    ap.add_argument("--depth", type=int, default=8, help="latent-recursion steps per round")
    ap.add_argument("--topk-elite", type=int, default=8)
    ap.add_argument("--parents", type=int, default=4, help="parents aggregated per child")
    ap.add_argument("--temp", type=float, default=0.5, help="selection softmax temperature")
    ap.add_argument("--mutation", type=float, default=0.02, help="latent mutation noise std")
    ap.add_argument("--smoke", action="store_true", help="random weights + random data, just test the loop")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    if device.type != "cuda" and dtype != torch.float32:
        print(f"[note] {device.type} can be flaky in {args.dtype}; float32 is safest.")
    print(f"[info] device={device} dtype={args.dtype} pop={args.pop} rounds={args.rounds} depth={args.depth}")

    if args.smoke:
        meta = {"seq_len": 81, "vocab_size": 11, "num_puzzle_identifiers": 1, "ignore_label_id": None, "sets": ["all"]}
        puzzles = make_smoke_puzzles(meta, args.num_puzzles)
    else:
        if not args.data_dir:
            sys.exit("Provide --data-dir (or use --smoke). See module docstring.")
        meta = read_metadata(args.data_dir, args.split)
        puzzles = load_puzzles(args.data_dir, args.split, meta, args.num_puzzles)

    arch_cfg = load_arch_config(args.arch_config)
    model = build_model(arch_cfg, meta, args.pop, device, dtype)
    if args.checkpoint:
        n_loaded, n_missing = load_checkpoint(model, args.checkpoint)
        print(f"[info] loaded {n_loaded} tensors from checkpoint ({n_missing} missing)")
    elif not args.smoke:
        print("[warn] no --checkpoint: running with RANDOM weights (accuracy will be ~0).")

    tallies: Dict[str, int] = {}
    for i, (inp, pid, lab) in enumerate(puzzles):
        res = rsa_solve(model, inp, pid, lab, args, device, dtype)
        for k, v in res.items():
            tallies[k] = tallies.get(k, 0) + int(v)
        if (i + 1) % 8 == 0 or i + 1 == len(puzzles):
            line = "  ".join(f"{k}={tallies[k]}/{i + 1}" for k in sorted(tallies))
            print(f"[{i + 1}/{len(puzzles)}] {line}")

    n = len(puzzles)
    print("\n=== exact-match accuracy ===")
    for k in sorted(tallies):
        print(f"  {k:20s} {tallies[k] / n:.4f}  ({tallies[k]}/{n})")


if __name__ == "__main__":
    main()
