"""
prepare_data.py
---------------
Pre-samples training data ONCE and saves to disk.
All conditions load the same fixed dataset — ensures fairness and reproducibility.

Run ONCE before any condition training:
    python prepare_data.py --max_per_dataset 10000 --datasets medqa medmcqa pubmedqa
"""

import os
import json
import argparse
import random
import sys

sys.path.insert(0, '.')
from data_loader import FMACADataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_dataset", type=int, default=10000)
    parser.add_argument("--datasets", nargs="+", default=["medqa", "medmcqa", "pubmedqa"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./shared_train_data.json")
    args = parser.parse_args()

    random.seed(args.seed)

    loader = FMACADataLoader()
    all_data = loader.load_all(
        split="train",
        datasets=args.datasets,
        max_per_dataset=args.max_per_dataset,
    )

    # Filter invalid
    all_data = [d for d in all_data if d.get("answer", "?") != "?"]

    # Shuffle with fixed seed
    random.shuffle(all_data)

    with open(args.output, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"Saved {len(all_data)} examples to {args.output}")
    print("Dataset breakdown:")

    from collections import Counter
    sources = Counter(d["source"] for d in all_data)
    tasks = Counter(d["task"] for d in all_data)

    for k, v in sources.items():
        print(f"  {k}: {v}")

    for k, v in tasks.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
