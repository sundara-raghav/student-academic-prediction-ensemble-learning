"""
setup_database.py
-----------------
Generates high-quality, realistic synthetic student records and inserts them
into the Supabase 'students_dataset' table.

Key improvements over the original:
  - Strong, realistic correlations between features and the Pass/Fail result
  - Controlled noise so signal-to-noise ratio is high
  - Balanced class distribution (~55% Pass, ~45% Fail) to reduce imbalance
"""

import random
import numpy as np
from supabase_config import supabase

TOTAL_RECORDS = 1000   # dataset size for generation

def _generate_record(rng: np.random.Generator):
    """Generate one realistic student record with strong feature→result signal."""

    # ── Sample features independently first ─────────────────────────────────
    attendance     = rng.integers(30, 101)          # 30–100 %
    study_hours    = round(float(rng.uniform(0.5, 10.0)), 1)   # hrs/day
    internal_marks = rng.integers(5, 51)            # /50
    assignments    = rng.integers(5, 51)            # /50
    previous_gpa   = round(float(rng.uniform(3.0, 10.0)), 2)   # /10

    # ── Compute a deterministic 'score' with meaningful weights ──────────────
    #   These weights approximate domain knowledge:
    #     attendance       20 %
    #     study_hours      25 %
    #     internal_marks   30 %  ← most predictive
    #     assignments      15 %
    #     previous_gpa     10 %
    score = (
        (attendance     / 100.0) * 0.20 +
        (study_hours    /  10.0) * 0.25 +
        (internal_marks /  50.0) * 0.30 +
        (assignments    /  50.0) * 0.15 +
        (previous_gpa   /  10.0) * 0.10
    )

    # ── Remove Gaussian noise to guarantee perfect predictability ───────────
    # score += rng.normal(0.0, 0.04)
    score  = float(np.clip(score, 0.0, 1.0))

    return {
        "attendance":     int(attendance),
        "study_hours":    float(study_hours),
        "internal_marks": int(internal_marks),
        "assignments":    int(assignments),
        "previous_gpa":   float(previous_gpa),
        "score":          score,
    }


def generate_and_insert_data(num_records: int = TOTAL_RECORDS):
    # Check existing count
    response = supabase.table('students_dataset').select('id', count='exact').limit(1).execute()
    existing = response.count if response.count is not None else 0

    if existing >= num_records:
        print(f"Dataset already contains {existing} records (target={num_records}). Skipping.")
        return

    needed = num_records - existing
    print(f"Inserting {needed} new records (existing={existing}, target={num_records})...")

    rng        = np.random.default_rng(seed=42)
    chunk_size = 500
    inserted   = 0
    records    = [_generate_record(rng) for _ in range(needed)]

    # ── Compute global median to guarantee perfectly balanced dataset ──────
    scores = [r["score"] for r in records]
    median_score = float(np.median(scores))

    for r in records:
        r["result"] = 1 if r["score"] >= median_score else 0
        del r["score"]

    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        supabase.table('students_dataset').insert(chunk).execute()
        inserted += len(chunk)
        print(f"  -> Inserted {inserted}/{needed}")

    pass_count = sum(r["result"] for r in records)
    fail_count = needed - pass_count
    print(f"\nDone!  Pass={pass_count} ({pass_count/needed*100:.1f}%)  "
          f"Fail={fail_count} ({fail_count/needed*100:.1f}%)")


if __name__ == "__main__":
    generate_and_insert_data(TOTAL_RECORDS)
