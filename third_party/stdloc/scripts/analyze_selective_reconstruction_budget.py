import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.selective_reconstruction import summarize_locability_selection


def _load_scores(score_file):
    with open(score_file, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        if "score_avg" in payload:
            return torch.as_tensor(payload["score_avg"], dtype=torch.float32)
        if "sampled_scores" in payload:
            return torch.as_tensor(payload["sampled_scores"], dtype=torch.float32)
    return torch.as_tensor(payload, dtype=torch.float32)


def _parse_ratios(value):
    return [float(item) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--top_ratios", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--min_weight", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    scores = _load_scores(args.score_file)
    rows = summarize_locability_selection(
        scores,
        top_ratios=_parse_ratios(args.top_ratios),
        min_weight=args.min_weight,
        gamma=args.gamma,
    )
    payload = {
        "score_file": str(Path(args.score_file)),
        "num_scores": int(torch.as_tensor(scores).numel()),
        "min_weight": args.min_weight,
        "gamma": args.gamma,
        "rows": rows,
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
