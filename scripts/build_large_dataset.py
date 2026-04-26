#!/usr/bin/env python3
"""
Large dataset builder for Email Triage GRPO training.

Generates diverse training examples by:
1. Running all seeds 0..N across all task levels
2. Compositing email templates with randomised metadata
3. Covering all (category × priority × department) combinations

Usage:
    python scripts/build_large_dataset.py --output data/train.jsonl --seeds 50
    python scripts/build_large_dataset.py --output data/train.jsonl --seeds 200 --tasks hard
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.email_generator import generate_emails
from server.tasks import get_task, list_task_ids


def build_large_dataset(
    task_ids=None,
    num_seeds: int = 100,
    seed_offset: int = 0,
    output_path: str = "data/train.jsonl",
    deduplicate: bool = True,
):
    """
    Generate a large dataset by running multiple seeds across task levels.

    Each seed produces a different shuffled inbox, giving diverse orderings
    of the same underlying email templates — the model learns to handle any
    ordering without overfitting to a fixed sequence.

    Args:
        task_ids:     which tasks to include (default: all)
        num_seeds:    how many seeds per task to run
        seed_offset:  start seed from this value (useful for val splits)
        output_path:  where to write the JSONL file
        deduplicate:  remove duplicate prompts (same email_id, different seed)
    """
    if task_ids is None:
        task_ids = list_task_ids()  # ['easy', 'medium', 'hard']

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    seen_bodies = set()
    examples = []
    skipped = 0

    for task_id in task_ids:
        task = get_task(task_id)
        print(f"\n[{task_id.upper()}] generating {num_seeds} seeds × {_task_size(task_id)} emails "
              f"= up to {num_seeds * _task_size(task_id)} examples...")

        for seed in range(seed_offset, seed_offset + num_seeds):
            emails, ground_truths = generate_emails(task_id, seed)
            truth_map = {gt.email_id: gt for gt in ground_truths}

            for email in emails:
                gt = truth_map[email.email_id]

                # Dedup by (category, body snippet) — avoids pure-duplicate prompts
                # while keeping same email at different seeds (different email_id)
                dedup_key = (gt.category, email.body[:80].strip())
                if deduplicate and dedup_key in seen_bodies:
                    skipped += 1
                    continue
                seen_bodies.add(dedup_key)

                prompt = _format_prompt(email.model_dump(), task.description)

                examples.append({
                    # ── GRPO fields ──────────────────────────────────────
                    "prompt": prompt,  # str (not chat list) for JSONL storage
                    # ── Ground truth metadata for reward functions ───────
                    "email_id":              email.email_id,
                    "task_id":               task_id,
                    "seed":                  seed,
                    "gt_category":           gt.category,
                    "gt_priority":           gt.priority,
                    "gt_department":         gt.department,
                    "gt_requires_response":  gt.requires_response,
                    "gt_expected_keywords":  gt.expected_keywords or [],
                    "gt_should_escalate":    getattr(gt, "should_escalate", False),
                    # ── Task metadata ─────────────────────────────────────
                    "requires_priority":     task.requires_priority,
                    "requires_routing":      task.requires_routing,
                    "requires_response":     task.requires_response,
                    "curriculum_level":      task.curriculum_level,
                })

        count_for_task = sum(1 for e in examples if e["task_id"] == task_id)
        print(f"  → {count_for_task} unique examples (skipped {skipped} dups)")
        skipped = 0

    # ── Category distribution stats ──────────────────────────────────────────
    cats = {}
    for ex in examples:
        cats[ex["gt_category"]] = cats.get(ex["gt_category"], 0) + 1

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(examples)} examples")
    print(f"Distribution: {json.dumps(cats, indent=2)}")

    # ── Write JSONL ──────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved → {output_path}")
    return examples


def _task_size(task_id: str) -> int:
    return {"easy": 5, "medium": 10, "hard": 20}.get(task_id, 10)


def _format_prompt(email: dict, task_description: str) -> str:
    """Format email into the user prompt string (same as train.py)."""
    lines = [
        f"TASK: {task_description}",
        "",
        f"From: {email.get('sender_name', '')} <{email.get('sender', '')}>",
        f"Subject: {email.get('subject', '')}",
        f"Time: {email.get('timestamp', '')}",
        f"Has attachment: {'Yes' if email.get('has_attachment') else 'No'}",
        f"Is reply: {'Yes' if email.get('is_reply') else 'No'}",
    ]
    if email.get("thread_id"):
        lines.append(f"Thread: {email['thread_id']}")
    lines += ["", "---", "", email.get("body", ""), "", "---",
              "", f"Email ID: {email['email_id']}"]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build large GRPO training dataset")
    parser.add_argument("--output",  default="data/train.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--seeds",   type=int, default=100,
                        help="Number of seeds per task (default: 100)")
    parser.add_argument("--offset",  type=int, default=0,
                        help="Seed offset (use 900+ for val split)")
    parser.add_argument("--tasks",   default="all",
                        help="Comma-separated task ids or 'all' (default: all)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable deduplication (more examples, more overlap)")
    args = parser.parse_args()

    task_ids = None if args.tasks == "all" else args.tasks.split(",")
    build_large_dataset(
        task_ids=task_ids,
        num_seeds=args.seeds,
        seed_offset=args.offset,
        output_path=args.output,
        deduplicate=not args.no_dedup,
    )


if __name__ == "__main__":
    main()
