"""
EDA for the Business Entity Resolution challenge.

Usage:
    pip install pandas numpy rapidfuzz matplotlib
    python eda_entity_resolution.py --data-dir dataset

Expects the standard layout:
    dataset/train/train_source1.tsv
    dataset/train/train_source2.tsv
    dataset/train/train_source3.tsv
    dataset/train/train_ground_truth.tsv
    dataset/test/test_source1.tsv
    dataset/test/test_source2.tsv
    dataset/test/test_source3.tsv

Prints profiling stats to stdout and (if matplotlib + rapidfuzz are present)
saves similarity_separation.png next to this script.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False
    print("[warn] rapidfuzz not installed (pip install rapidfuzz) -> skipping similarity-separation section")


def load(path: Path) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: entity IDs and codes must not be
    # silently coerced to numbers/NaN, and "NA" as a country/text token
    # must not be read as a missing value.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def basic_profile(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    for col in df.columns:
        n_empty = (df[col].astype(str).str.strip() == "").sum()
        n_unique = df[col].nunique()
        print(f"  {col:20s} empty={n_empty:6d} ({n_empty/len(df):5.1%})  unique={n_unique}")


def duplicate_check(name: str, df: pd.DataFrame, key_cols=("business_name", "business_address")) -> None:
    key_cols = [c for c in key_cols if c in df.columns]
    dupe_mask = df.duplicated(subset=key_cols, keep=False)
    print(f"{name}: exact duplicate rows on {key_cols}: {dupe_mask.sum()} ({dupe_mask.mean():.2%})")


def country_breakdown(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name} country distribution:")
    print(df["country"].value_counts(dropna=False).to_string())


def length_stats(name: str, df: pd.DataFrame) -> None:
    name_len = df["business_name"].astype(str).str.len()
    addr_len = df["business_address"].astype(str).str.len()
    print(f"\n{name} business_name length: mean={name_len.mean():.1f} median={name_len.median():.0f} max={name_len.max()}")
    print(f"{name} business_address length: mean={addr_len.mean():.1f} median={addr_len.median():.0f} max={addr_len.max()}")
    print(f"{name} rows with empty address: {(addr_len == 0).sum()} ({(addr_len == 0).mean():.1%})")


LEGAL_SUFFIXES = [
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "pvt", "private", "plc", "pte", "gmbh", "sa",
]


def suffix_frequency(name: str, df: pd.DataFrame) -> None:
    text = df["business_name"].astype(str).str.lower()
    print(f"\n{name} legal-suffix token frequency (of {len(df)} rows):")
    for suf in LEGAL_SUFFIXES:
        pattern = rf"\b{re.escape(suf)}\b\.?"
        count = text.str.contains(pattern, regex=True, na=False).sum()
        if count:
            print(f"  {suf:15s} {count:6d} ({count/len(df):5.1%})")


def parse_ids(cell: str):
    cell = (cell or "").strip()
    return [] if cell == "" else [x.strip() for x in cell.split(",")]


def ground_truth_profile(gt: pd.DataFrame, s2_ids: set, s3_ids: set) -> pd.DataFrame:
    print("\n=== ground truth ===")
    print("source1 entities with a GT row:", len(gt))

    gt = gt.copy()
    gt["match_list"] = gt["matched_entity_ids"].apply(parse_ids)
    gt["n_matches"] = gt["match_list"].apply(len)

    n_singleton = (gt["n_matches"] == 0).sum()
    print(f"singletons (0 matches): {n_singleton} ({n_singleton/len(gt):.1%})")
    print("match-count distribution:\n", gt["n_matches"].value_counts().sort_index().to_string())

    all_ids = [i for lst in gt["match_list"] for i in lst]
    n_s2 = sum(1 for i in all_ids if i.startswith("S2-"))
    n_s3 = sum(1 for i in all_ids if i.startswith("S3-"))
    print(f"total matched IDs: {len(all_ids)}  (from S2: {n_s2}, from S3: {n_s3})")

    bad = [i for i in all_ids if not (i in s2_ids or i in s3_ids)]
    print("matched IDs NOT found in source2/source3 files (should be 0):", len(bad))
    if bad[:5]:
        print("  examples:", bad[:5])

    dupe_rows = gt[gt["match_list"].apply(lambda lst: len(lst) != len(set(lst)))]
    print("GT rows with an internal duplicate ID (should be 0):", len(dupe_rows))

    # Is a given S2/S3 record ever claimed by more than one S1 entity?
    # (tells you whether "matched_entity_ids" is a strict partition or can overlap)
    owner_count = {}
    for _, row in gt.iterrows():
        for m in row["match_list"]:
            owner_count[m] = owner_count.get(m, 0) + 1
    n_shared = sum(1 for v in owner_count.values() if v > 1)
    print(f"S2/S3 records matched to MORE THAN ONE S1 entity: {n_shared}")

    return gt


def similarity_separation(train_s1, train_s2, train_s3, gt, n_pos=400, n_neg=400, seed=0):
    """Sanity check: do string-similarity scores actually separate true
    matches from random non-matches? This tells you whether classic fuzzy
    matching features alone will carry the model, or whether you need
    semantic embeddings to close the gap."""
    if not HAVE_RAPIDFUZZ:
        return

    rng = np.random.default_rng(seed)
    s1_map = train_s1.set_index("entity_id")
    s2_map = train_s2.set_index("entity_id")
    s3_map = train_s3.set_index("entity_id")

    def get_row(eid):
        return s2_map.loc[eid] if eid.startswith("S2-") else s3_map.loc[eid]

    pos_pairs = [(r["source1_entity_id"], m) for _, r in gt.iterrows() for m in r["match_list"]]
    rng.shuffle(pos_pairs)
    pos_pairs = pos_pairs[:n_pos]

    all_other_ids = list(train_s2["entity_id"]) + list(train_s3["entity_id"])
    gt_lookup = {r["source1_entity_id"]: set(r["match_list"]) for _, r in gt.iterrows()}
    s1_ids = list(train_s1["entity_id"])
    neg_pairs = []
    while len(neg_pairs) < n_neg:
        s1_id = rng.choice(s1_ids)
        cand = rng.choice(all_other_ids)
        if cand not in gt_lookup.get(s1_id, set()):
            neg_pairs.append((s1_id, cand))

    def scores(pairs):
        name_s, addr_s = [], []
        for s1_id, other_id in pairs:
            r1, r2 = s1_map.loc[s1_id], get_row(other_id)
            name_s.append(fuzz.token_sort_ratio(str(r1["business_name"]), str(r2["business_name"])))
            addr_s.append(fuzz.token_sort_ratio(str(r1["business_address"]), str(r2["business_address"])))
        return np.array(name_s), np.array(addr_s)

    pos_name, pos_addr = scores(pos_pairs)
    neg_name, neg_addr = scores(neg_pairs)

    print("\n=== token_sort_ratio separation, positives vs. random negatives (0-100) ===")
    print(f"positive  name: mean={pos_name.mean():5.1f}  p10={np.percentile(pos_name,10):5.1f}")
    print(f"negative  name: mean={neg_name.mean():5.1f}  p90={np.percentile(neg_name,90):5.1f}")
    print(f"positive  addr: mean={pos_addr.mean():5.1f}  p10={np.percentile(pos_addr,10):5.1f}")
    print(f"negative  addr: mean={neg_addr.mean():5.1f}  p90={np.percentile(neg_addr,90):5.1f}")
    print("-> if positive-p10 << negative-p90 for both, a single global threshold will")
    print("   struggle and you need a learned model / more features, not just a cutoff.")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(pos_name, bins=20, alpha=0.6, label="match")
        axes[0].hist(neg_name, bins=20, alpha=0.6, label="non-match")
        axes[0].set_title("name token_sort_ratio")
        axes[0].legend()
        axes[1].hist(pos_addr, bins=20, alpha=0.6, label="match")
        axes[1].hist(neg_addr, bins=20, alpha=0.6, label="non-match")
        axes[1].set_title("address token_sort_ratio")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig("similarity_separation.png", dpi=120)
        print("saved similarity_separation.png")
    except ImportError:
        pass


def country_consistency_check(train_s1, train_s2, train_s3, gt):
    """Critical check before hard-partitioning blocking by country: what
    fraction of TRUE matches actually share a country label across sources?
    If this isn't ~100%, country partitioning silently caps your recall
    ceiling by exactly that gap -- decide if that trade-off is worth the
    (very large) compute savings before committing to it."""
    s1_map = train_s1.set_index("entity_id")["country"]
    s2_map = train_s2.set_index("entity_id")["country"]
    s3_map = train_s3.set_index("entity_id")["country"]

    total, mismatched = 0, 0
    examples = []
    for _, row in gt.iterrows():
        cell = (row["matched_entity_ids"] or "").strip()
        if not cell:
            continue
        c1 = s1_map.get(row["source1_entity_id"])
        for m in cell.split(","):
            m = m.strip()
            c2 = s2_map.get(m) if m.startswith("S2-") else s3_map.get(m)
            total += 1
            if c1 != c2:
                mismatched += 1
                if len(examples) < 5:
                    examples.append((row["source1_entity_id"], c1, m, c2))

    print(f"\n=== country consistency across true matches ===")
    print(f"true pairs checked: {total}, country mismatches: {mismatched} ({mismatched/max(total,1):.2%})")
    if examples:
        print("example mismatches (source1_id, c1, matched_id, c2):")
        for ex in examples:
            print(" ", ex)
    print("-> this percentage is the recall you'd permanently lose by hard-partitioning blocking by country.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    args = ap.parse_args()
    root = Path(args.data_dir)

    train_s1 = load(root / "train" / "train_source1.tsv")
    train_s2 = load(root / "train" / "train_source2.tsv")
    train_s3 = load(root / "train" / "train_source3.tsv")
    gt = load(root / "train" / "train_ground_truth.tsv")
    test_s1 = load(root / "test" / "test_source1.tsv")
    test_s2 = load(root / "test" / "test_source2.tsv")
    test_s3 = load(root / "test" / "test_source3.tsv")

    for name, df in [("train_source1", train_s1), ("train_source2", train_s2),
                      ("train_source3", train_s3), ("test_source1", test_s1),
                      ("test_source2", test_s2), ("test_source3", test_s3)]:
        basic_profile(name, df)
        duplicate_check(name, df)
        country_breakdown(name, df)
        length_stats(name, df)
        suffix_frequency(name, df)

    print("\n--- country coverage check ---")
    train_countries = set(train_s1["country"]) | set(train_s2["country"]) | set(train_s3["country"])
    test_countries = set(test_s1["country"]) | set(test_s2["country"]) | set(test_s3["country"])
    print("countries only in test (expect France):", test_countries - train_countries)
    print("countries only in train:", train_countries - test_countries)

    s2_ids, s3_ids = set(train_s2["entity_id"]), set(train_s3["entity_id"])
    gt = ground_truth_profile(gt, s2_ids, s3_ids)

    country_consistency_check(train_s1, train_s2, train_s3, gt)
    similarity_separation(train_s1, train_s2, train_s3, gt)


if __name__ == "__main__":
    main()
