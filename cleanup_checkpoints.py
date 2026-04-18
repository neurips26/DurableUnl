"""
cleanup_checkpoints.py
======================
Delete checkpoints before re-running training.

Usage:
  # Delete one specific method's checkpoint:
  python cleanup_checkpoints.py --methods ga

  # Delete multiple:
  python cleanup_checkpoints.py --methods ga npo scrub

  # Delete ALL checkpoints (wipe everything):
  python cleanup_checkpoints.py --all

  # Just list what checkpoints exist (dry run):
  python cleanup_checkpoints.py --list
"""

import argparse
import json
import os
import shutil
import sys


def list_checkpoints(ckpt_dir: str):
    """Print all existing checkpoints with their saved metrics."""
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory '{ckpt_dir}' does not exist yet.")
        return []

    found = []
    for name in sorted(os.listdir(ckpt_dir)):
        result_json = os.path.join(ckpt_dir, name, "result.json")
        model_dir   = os.path.join(ckpt_dir, name, "model")
        if os.path.isdir(os.path.join(ckpt_dir, name)):
            found.append(name)
            has_result = os.path.exists(result_json)
            has_model  = os.path.exists(model_dir)

            print(f"\n  [{name}]")
            print(f"    model/    : {'✓ present' if has_model  else '✗ missing'}")
            print(f"    result.json: {'✓ present' if has_result else '✗ missing'}")

            if has_result:
                try:
                    with open(result_json) as f:
                        data = json.load(f)
                    m    = data.get("metrics", {})
                    ts   = data.get("saved_at", "unknown time")
                    fa   = m.get("forget_acc", "?")
                    ra   = m.get("retain_acc", "?")
                    qi4  = m.get("quant_int4", "?")
                    wt   = m.get("wall_time_min", "?")
                    print(f"    Saved at  : {ts}")
                    print(f"    forget_acc: {fa}  |  retain_acc: {ra}")
                    print(f"    quant_int4: {qi4} |  time: {wt} min")
                except Exception as e:
                    print(f"    (could not read result.json: {e})")

    if not found:
        print("  No checkpoints found.")
    return found


def delete_checkpoint(ckpt_dir: str, method_name: str, dry_run: bool = False):
    """Delete the checkpoint directory for one method."""
    path = os.path.join(ckpt_dir, method_name)
    if not os.path.exists(path):
        print(f"  [{method_name}] No checkpoint found at '{path}' — nothing to delete.")
        return False

    if dry_run:
        print(f"  [{method_name}] Would delete: {path}")
        return True

    shutil.rmtree(path)
    print(f"  [{method_name}] Deleted: {path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Manage DurableUn checkpoints")
    parser.add_argument(
        "--methods", nargs="+",
        help="Method names to delete (e.g. --methods ga npo scrub)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Delete ALL checkpoints"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all checkpoints without deleting"
    )
    parser.add_argument(
        "--ckpt_dir", default="checkpoints",
        help="Checkpoint directory (default: checkpoints)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    args = parser.parse_args()

    print(f"\n=== DurableUn Checkpoint Manager ===")
    print(f"Checkpoint directory: {os.path.abspath(args.ckpt_dir)}\n")

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list or (not args.all and not args.methods):
        print("Existing checkpoints:")
        list_checkpoints(args.ckpt_dir)
        return

    # ── Confirm before deleting ───────────────────────────────────────────────
    if args.all:
        existing = list_checkpoints(args.ckpt_dir)
        if not existing:
            return
        print(f"\n{'[DRY RUN] Would delete' if args.dry_run else 'About to delete'} "
              f"ALL {len(existing)} checkpoints: {existing}")
        if not args.dry_run:
            confirm = input("Type 'yes' to confirm: ").strip().lower()
            if confirm != "yes":
                print("Cancelled.")
                return
        for name in existing:
            delete_checkpoint(args.ckpt_dir, name, args.dry_run)

    elif args.methods:
        print(f"{'[DRY RUN] Would delete' if args.dry_run else 'Deleting'}: {args.methods}")
        if not args.dry_run:
            confirm = input(f"Delete checkpoints for {args.methods}? Type 'yes': ").strip().lower()
            if confirm != "yes":
                print("Cancelled.")
                return
        for name in args.methods:
            delete_checkpoint(args.ckpt_dir, name, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
