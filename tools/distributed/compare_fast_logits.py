#!/usr/bin/env python3
"""Compare same-pack fast TP logits under a policy frozen before the run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(reference: np.ndarray, candidate: np.ndarray, policy: dict) -> dict:
    if reference.ndim != 2 or candidate.shape != reference.shape or not reference.size:
        raise ValueError("logits must be nonempty matching [positions,vocabulary] arrays")
    if reference.shape[1] < 2:
        raise ValueError("vocabulary must contain at least two tokens")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("nonfinite logits cannot qualify")
    limits = policy["logits"]
    rows = []
    for a, b in zip(reference, candidate):
        a, b = a.astype(np.float64), b.astype(np.float64)
        a -= a.max()
        b -= b.max()
        logp = a - np.log(np.exp(a).sum())
        logq = b - np.log(np.exp(b).sum())
        kl = max(0.0, float(np.dot(np.exp(logp), logp - logq)))
        largest = np.partition(a, -2)[-2:]
        margin = float(largest.max() - largest.min())
        rows.append({"kl": kl, "top1_reference": int(a.argmax()), "top1_candidate": int(b.argmax()),
                     "top1_agrees": bool(a.argmax() == b.argmax()), "reference_margin": margin,
                     "near_tie": margin <= limits["near_tie_logit_margin"]})
    mean_kl = float(np.mean([r["kl"] for r in rows]))
    maximum = max(r["kl"] for r in rows)
    agreement = sum(r["top1_agrees"] for r in rows) / len(rows)
    stable = [r for r in rows if not r["near_tie"]]
    return {"qualified": mean_kl <= limits["kl_mean_max"] and maximum <= limits["kl_max_max"] and
                         agreement >= limits["top1_min_fraction"],
            "positions": len(rows), "vocabulary": reference.shape[1], "kl_mean": mean_kl,
            "kl_max": maximum, "top1_agreement": agreement,
            "stable_margin_top1_agreement": None if not stable else
                sum(r["top1_agrees"] for r in stable) / len(stable),
            "rows": rows, "scope": "same-pack logits; storage fidelity, state isolation and task quality are separate gates"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.vocabulary < 2:
            raise ValueError("invalid vocabulary")
        reference = np.fromfile(args.reference, dtype="<f4").reshape(-1, args.vocabulary)
        candidate = np.fromfile(args.candidate, dtype="<f4").reshape(-1, args.vocabulary)
        policy = json.loads(args.policy.read_text())
        result = compare(reference, candidate, policy)
        result["files_sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for key, path in [("reference", args.reference), ("candidate", args.candidate),
                                                    ("policy", args.policy)]}
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
        return 0 if result["qualified"] else 1
    except (ValueError, KeyError, OSError) as exc:
        print("fast logits comparison refused:", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
