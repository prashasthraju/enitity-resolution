"""
Business Entity Resolution -- scalable baseline (v2).

Rebuilt around real EDA findings: ~2.2M / 5.0M / 5.3M rows (brute-force
all-pairs blocking is not an option), test adds a country (France) unseen
in train, some Indian addresses are in Telugu/Kannada script, ~3.3% of
S2/S3 addresses are missing.

Stages:
    1. prepare()              -- country-aware normalization: romanize any
                                  script to Latin (also strips French accents
                                  for free), expand country-specific street/
                                  legal abbreviations, extract postal codes
    2. build_candidates_all() -- blocking, scales as a hash-join not n*m:
                                  country partition -> token inverted index
                                  (name AND address) + exact postal-code match
                                  + short-prefix safety net -> ranked, capped
                                  per S1 entity. This union is candidate_pairs.tsv
    3. featurize_candidates() -- rapidfuzz + token/char-ngram/digit overlap +
                                  postal/country match, via O(1) dict lookups
                                  (no per-row pandas .loc -- that's what makes
                                  the v1 version too slow at this scale)
    4. train_classifier()     -- LightGBM binary classifier (falls back to
                                  sklearn HGB), country is NEVER a raw model
                                  feature (France has zero training examples --
                                  only country_match, which generalizes)
    5. tune_threshold()       -- sweeps the decision threshold against the
                                  real macro F0.5 formula (fbeta_eval.py) --
                                  this, not the training loss, is where the
                                  precision-heavy behavior comes from
    6. predict / write_output

Usage:
    pip install pandas numpy scikit-learn rapidfuzz lightgbm unidecode
    python baseline_pipeline.py --data-dir dataset --out-dir output

Not run against your real data (none is available in this session) --
validated end-to-end against a synthetic multi-country, multi-script,
~100k-row dataset instead. See the note at the bottom of this file for
measured throughput and how to extrapolate to your actual row counts.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from fbeta_eval import score_submission

try:
    from rapidfuzz import fuzz
except ImportError as e:
    raise SystemExit("pip install rapidfuzz") from e

try:
    from unidecode import unidecode
except ImportError as e:
    raise SystemExit("pip install unidecode") from e

try:
    import lightgbm as lgb
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier


# --------------------------------------------------------------------------
# 1. Country-aware normalization
# --------------------------------------------------------------------------

# Deliberately partial lists -- extend these from what your EDA's
# suffix_frequency() / manual spot-checks turn up for each country.
STREET_ABBR = {
    "us": {"rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard", "ln": "lane",
           "dr": "drive", "apt": "apartment", "ste": "suite", "hwy": "highway", "ct": "court", "pl": "place"},
    "india": {"rd": "road", "st": "street", "marg": "road", "nr": "near", "opp": "opposite",
              "soc": "society", "apt": "apartment", "colony": "colony", "nagar": "nagar"},
    "france": {"r": "rue", "av": "avenue", "bd": "boulevard", "imp": "impasse", "che": "chemin",
               "pl": "place", "all": "allee", "fg": "faubourg", "rte": "route"},
}
LEGAL_SUFFIX = {
    "us": {"corp": "corporation", "co": "company", "inc": "incorporated", "llc": "llc", "ltd": "limited"},
    "india": {"corp": "corporation", "co": "company", "pvt": "private", "ltd": "limited", "llp": "llp"},
    # French legal forms (sarl/sas/sa/eurl/snc) are almost always used as-is,
    # not spelled out -- left untouched rather than guessing an expansion.
    "france": {},
}
POSTAL_RE = {
    "us": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    "india": re.compile(r"\b\d{6}\b"),
    "france": re.compile(r"\b\d{5}\b"),
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def romanize(s: str) -> str:
    """No-op on plain ASCII; strips accents from French text; phonetically
    romanizes Telugu/Kannada/Devanagari/etc. so cross-script records still
    share tokens and character n-grams instead of having zero overlap."""
    return s if s.isascii() else unidecode(s)


def _normalize_with_table(s: str, table: dict) -> str:
    s = romanize(s or "").lower()
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    tokens = [table.get(t, t) for t in s.split()]
    return _SPACE_RE.sub(" ", " ".join(tokens)).strip()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """One-time preprocessing pass: adds _norm_name, _norm_addr, _postal.
    Groups by country so each country's abbreviation table is looked up
    once per group rather than once per row."""
    parts = []
    for country, grp in df.groupby("country", sort=False):
        grp = grp.copy()
        key = str(country).strip().lower()
        suffix_table = LEGAL_SUFFIX.get(key, {})
        street_table = STREET_ABBR.get(key, {})
        postal_pat = POSTAL_RE.get(key)

        grp["_norm_name"] = grp["business_name"].map(lambda s: _normalize_with_table(s, suffix_table))
        grp["_norm_addr"] = grp["business_address"].map(lambda s: _normalize_with_table(s, street_table))
        if postal_pat is not None:
            grp["_postal"] = grp["business_address"].map(
                lambda s: (postal_pat.findall(s or "") or [None])[-1]
            )
        else:
            grp["_postal"] = None  # unrecognized country -> no postal-code blocking/feature for it
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# 2. Blocking (scales as a hash-join, not n*m)
# --------------------------------------------------------------------------

def token_block(s1: pd.DataFrame, other: pd.DataFrame, text_col: str,
                 min_token_len: int = 3, keep_top_n_per_record: int = 3, max_block_size: int = 2000) -> pd.DataFrame:
    """Inverted-index / overlap blocking, robust to skewed vocabularies (a
    handful of generic words -- 'traders', 'enterprises', 'company' -- covering
    a large share of records, which is common in real business names, not
    just a synthetic artifact). Rather than a single global document-frequency
    cutoff (which can zero out blocking entirely if most tokens are generic),
    each S1 record is joined on its own N *rarest* tokens, so it always gets
    a chance at candidates even in a low-diversity vocabulary. A hard
    max_block_size still drops pathologically common tokens (numbers, generic
    single words spanning huge fractions of a country) so no single token can
    blow up the join. This is a hash join, so it scales roughly linearly in
    row count rather than quadratically."""
    s1_tok = s1[["entity_id", "country"]].assign(_token=s1[text_col].str.split()).explode("_token")
    other_tok = other[["entity_id", "country"]].assign(_token=other[text_col].str.split()).explode("_token")
    s1_tok = s1_tok[s1_tok["_token"].str.len() >= min_token_len]
    other_tok = other_tok[other_tok["_token"].str.len() >= min_token_len]
    if other_tok.empty or s1_tok.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_id", "shared_tokens"])

    freq = other_tok.groupby(["country", "_token"]).size().rename("_df").reset_index()
    freq = freq[freq["_df"] <= max_block_size]  # drop pathologically common tokens outright
    if freq.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_id", "shared_tokens"])

    s1_tok = s1_tok.merge(freq, on=["country", "_token"], how="inner")  # a token absent from `other` can't block
    if s1_tok.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_id", "shared_tokens"])
    s1_tok = s1_tok.sort_values(["entity_id", "_df"])
    s1_tok["_rk"] = s1_tok.groupby("entity_id").cumcount()
    s1_tok = s1_tok[s1_tok["_rk"] < keep_top_n_per_record]

    other_tok = other_tok.merge(freq[["country", "_token"]], on=["country", "_token"])

    merged = s1_tok.merge(other_tok, on=["country", "_token"], suffixes=("_s1", "_other"))
    if merged.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_id", "shared_tokens"])
    scored = merged.groupby(["entity_id_s1", "entity_id_other"]).size().rename("shared_tokens").reset_index()
    return scored.rename(columns={"entity_id_s1": "source1_entity_id", "entity_id_other": "candidate_id"})


def _drop_oversized_keys(keyed: pd.DataFrame, key_col: str, max_block_size: int) -> pd.DataFrame:
    sizes = keyed.groupby(key_col).size()
    good = sizes[sizes <= max_block_size].index
    return keyed[keyed[key_col].isin(good)]


def postal_block(s1: pd.DataFrame, other: pd.DataFrame, max_block_size: int = 300) -> pd.DataFrame:
    """Exact postal/PIN/ZIP match, used as a blocking key ONLY when it's
    precise enough (max_block_size) to be worth the join -- a 6-digit Indian
    PIN code can legitimately cover thousands of businesses, and joining on
    it unconditionally is the same combinatorial-explosion risk as a generic
    name token. When a postal code is too coarse to block on, name/address
    token blocking still finds the pair, and postal_match still fires as a
    (very strong) feature in compute_pair_features regardless of whether it
    was used here."""
    a = s1[s1["_postal"].notna()][["entity_id", "country", "_postal"]]
    b = other[other["_postal"].notna()][["entity_id", "country", "_postal"]]
    if a.empty or b.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_id", "shared_tokens"])
    b = _drop_oversized_keys(b.assign(_key=b["country"] + "|" + b["_postal"]), "_key", max_block_size)
    if b.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_id", "shared_tokens"])
    merged = a.merge(b, on=["country", "_postal"], suffixes=("_s1", "_other"))
    out = merged.rename(columns={"entity_id_s1": "source1_entity_id", "entity_id_other": "candidate_id"})
    out["shared_tokens"] = 99  # sentinel: always ranks first
    return out[["source1_entity_id", "candidate_id", "shared_tokens"]]


def prefix_block(s1: pd.DataFrame, other: pd.DataFrame, prefix_len: int = 5, max_block_size: int = 500) -> pd.DataFrame:
    """Safety net for very short / sparsely-tokenized names that token_block
    might miss entirely. Same over-sized-block risk as postal_block when a
    prefix is generic (many real business names genuinely share a common
    opening word) -- capped the same way."""
    def key(df):
        return df["_norm_name"].str.replace(" ", "").str[:prefix_len] + "|" + df["country"].str.lower()
    a = s1.assign(_key=key(s1))[["entity_id", "_key"]]
    b = other.assign(_key=key(other))[["entity_id", "_key"]]
    b = _drop_oversized_keys(b, "_key", max_block_size)
    if b.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_id", "shared_tokens"])
    merged = a.merge(b, on="_key", suffixes=("_s1", "_other"))
    out = merged.rename(columns={"entity_id_s1": "source1_entity_id", "entity_id_other": "candidate_id"})
    out["shared_tokens"] = 1
    return out[["source1_entity_id", "candidate_id", "shared_tokens"]]


def build_candidates(s1: pd.DataFrame, other: pd.DataFrame, top_k_per_entity: int = 50) -> pd.DataFrame:
    name_cand = token_block(s1, other, text_col="_norm_name")
    addr_cand = token_block(s1, other, text_col="_norm_addr")
    postal_cand = postal_block(s1, other)
    prefix_cand = prefix_block(s1, other)

    all_cand = pd.concat([name_cand, addr_cand, postal_cand, prefix_cand], ignore_index=True)
    if all_cand.empty:
        return all_cand.assign(rank=[])
    all_cand = (all_cand.sort_values("shared_tokens", ascending=False)
                .drop_duplicates(subset=["source1_entity_id", "candidate_id"], keep="first"))
    all_cand["rank"] = all_cand.groupby("source1_entity_id")["shared_tokens"].rank(method="first", ascending=False)
    return all_cand[all_cand["rank"] <= top_k_per_entity].drop(columns="rank")


def build_candidates_all(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, top_k_per_entity: int = 50) -> pd.DataFrame:
    parts = []
    for other, tag in [(s2, "S2"), (s3, "S3")]:
        if len(other) == 0:
            continue
        c = build_candidates(s1, other, top_k_per_entity=top_k_per_entity)
        c["source"] = tag
        parts.append(c)
    cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["source1_entity_id", "candidate_id", "shared_tokens", "source"]
    )
    if cand.empty:
        return cand
    cand = cand.sort_values("shared_tokens", ascending=False)
    cand["rank"] = cand.groupby("source1_entity_id").cumcount()
    return cand[cand["rank"] < top_k_per_entity].drop(columns="rank")


# --------------------------------------------------------------------------
# 3. Feature engineering (O(1) dict lookups, not pandas .loc per row)
# --------------------------------------------------------------------------

FEATURE_NAMES = [
    "name_ratio", "name_partial", "name_token_sort", "name_token_set", "name_jaccard", "name_char3_jaccard",
    "addr_ratio", "addr_partial", "addr_token_sort", "addr_token_set", "addr_jaccard", "addr_char3_jaccard",
    "digit_jaccard", "postal_match", "country_match", "name_len_diff", "addr_len_diff", "addr_missing",
]


def char_ngrams(s: str, n: int = 3) -> set:
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0  # neutral, not 1.0 -- two records both missing a field is NOT a match signal
    return len(a & b) / len(a | b)


def build_record_cache(df: pd.DataFrame) -> dict:
    cache = {}
    for eid, name, addr, postal, country in zip(
        df["entity_id"], df["_norm_name"], df["_norm_addr"], df["_postal"], df["country"]
    ):
        cache[eid] = {
            "name": name, "addr": addr,
            "name_tok": set(name.split()), "addr_tok": set(addr.split()),
            "name_3g": char_ngrams(name), "addr_3g": char_ngrams(addr),
            "digits": set(re.findall(r"\d+", addr)),
            "postal": postal, "country": str(country).strip().lower(),
        }
    return cache


def compute_pair_features(a: dict, b: dict) -> list:
    addr_missing = 1.0 if (len(a["addr"]) == 0 or len(b["addr"]) == 0) else 0.0
    if addr_missing:
        # Don't let two blank addresses look like a perfect match: fuzz.ratio("","")
        # and jaccard(set(),set()) are otherwise both maximal for missing text.
        addr_ratio = addr_partial = addr_token_sort = addr_token_set = 0.0
        addr_jac = addr_3g_jac = digit_jac = 0.0
    else:
        addr_ratio = fuzz.ratio(a["addr"], b["addr"]) / 100.0
        addr_partial = fuzz.partial_ratio(a["addr"], b["addr"]) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(a["addr"], b["addr"]) / 100.0
        addr_token_set = fuzz.token_set_ratio(a["addr"], b["addr"]) / 100.0
        addr_jac = jaccard(a["addr_tok"], b["addr_tok"])
        addr_3g_jac = jaccard(a["addr_3g"], b["addr_3g"])
        digit_jac = jaccard(a["digits"], b["digits"])

    postal_match = 1.0 if (a["postal"] and b["postal"] and a["postal"] == b["postal"]) else 0.0

    return [
        fuzz.ratio(a["name"], b["name"]) / 100.0,
        fuzz.partial_ratio(a["name"], b["name"]) / 100.0,
        fuzz.token_sort_ratio(a["name"], b["name"]) / 100.0,
        fuzz.token_set_ratio(a["name"], b["name"]) / 100.0,
        jaccard(a["name_tok"], b["name_tok"]),
        jaccard(a["name_3g"], b["name_3g"]),
        addr_ratio, addr_partial, addr_token_sort, addr_token_set, addr_jac, addr_3g_jac,
        digit_jac, postal_match,
        1.0 if a["country"] == b["country"] else 0.0,
        abs(len(a["name"]) - len(b["name"])) / max(len(a["name"]), len(b["name"]), 1),
        abs(len(a["addr"]) - len(b["addr"])) / max(len(a["addr"]), len(b["addr"]), 1),
        addr_missing,
    ]


def featurize_candidates(cands: pd.DataFrame, s1: pd.DataFrame, other: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
    """Feature computation is independent per pair, so it parallelizes
    cleanly across cores (n_jobs > 1) once your candidate count is large
    enough to be worth the process-pool overhead. On a single core this is
    the dominant cost of the whole pipeline (measured ~35k pairs/sec on a
    constrained 1-vCPU sandbox) -- use n_jobs on real hardware."""
    if cands.empty:
        return cands.assign(**{f: [] for f in FEATURE_NAMES})
    used_s1 = s1[s1["entity_id"].isin(cands["source1_entity_id"])]
    used_other = other[other["entity_id"].isin(cands["candidate_id"])]
    s1_cache = build_record_cache(used_s1)
    other_cache = build_record_cache(used_other)
    pairs = list(zip(cands["source1_entity_id"], cands["candidate_id"]))

    if n_jobs <= 1 or len(pairs) < 200_000:
        rows = [compute_pair_features(s1_cache[a], other_cache[b]) for a, b in pairs]
    else:
        import concurrent.futures
        import math
        chunk_size = math.ceil(len(pairs) / (n_jobs * 4))
        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        rows = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_featurize_worker, initargs=(s1_cache, other_cache)
        ) as ex:
            for chunk_rows in ex.map(_featurize_chunk, chunks):
                rows.extend(chunk_rows)

    feat_df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return pd.concat([cands.reset_index(drop=True), feat_df], axis=1)


# module-level (picklable) worker helpers for the multiprocessing path above
_WORKER_S1_CACHE = None
_WORKER_OTHER_CACHE = None


def _init_featurize_worker(s1_cache, other_cache):
    global _WORKER_S1_CACHE, _WORKER_OTHER_CACHE
    _WORKER_S1_CACHE, _WORKER_OTHER_CACHE = s1_cache, other_cache


def _featurize_chunk(pairs):
    return [compute_pair_features(_WORKER_S1_CACHE[a], _WORKER_OTHER_CACHE[b]) for a, b in pairs]


# --------------------------------------------------------------------------
# 4-5. Labeling, training, threshold tuning
# (unchanged in spirit from v1 -- reproduced here so this file is standalone)
# --------------------------------------------------------------------------

def build_labeled_set(cands_feat: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    gt_pairs = set()
    for _, row in gt.iterrows():
        cell = (row["matched_entity_ids"] or "").strip()
        if cell:
            for m in cell.split(","):
                gt_pairs.add((row["source1_entity_id"], m.strip()))

    df = cands_feat.copy()
    df["label"] = [
        int((a, b) in gt_pairs) for a, b in zip(df["source1_entity_id"], df["candidate_id"])
    ]

    found = set(zip(df.loc[df["label"] == 1, "source1_entity_id"], df.loc[df["label"] == 1, "candidate_id"]))
    missed = gt_pairs - found
    if missed:
        print(f"[blocking] WARNING: {len(missed)}/{len(gt_pairs)} true pairs not in candidate set "
              f"(blocking recall = {1 - len(missed)/max(len(gt_pairs),1):.1%}). "
              f"Raise top_k_per_entity, max_block_size, or max_df_ratio.")
    else:
        print(f"[blocking] recall = 100% on {len(gt_pairs)} true pairs")
    return df


def train_classifier(train_df: pd.DataFrame):
    X, y = train_df[FEATURE_NAMES], train_df["label"]
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    if HAVE_LGBM:
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=n_neg / max(n_pos, 1), random_state=0, verbosity=-1,
        )
        model.fit(X, y)
    else:
        model = HistGradientBoostingClassifier(random_state=0)
        model.fit(X, y, sample_weight=np.where(y == 1, n_neg / max(n_pos, 1), 1.0))
    return model


def predictions_to_df(cands_scored: pd.DataFrame, threshold: float, all_s1_ids, min_addr_sim_if_present: float = 0.15) -> pd.DataFrame:
    """Threshold + a business-rule guard: don't accept a match on name alone
    when the address is present but dissimilar (classic chain/franchise false
    merge). When address is missing on either side, this guard is skipped --
    the model must decide from name + postal alone in that case."""
    guard_ok = (cands_scored["addr_missing"] == 1.0) | (cands_scored["addr_token_sort"] >= min_addr_sim_if_present)
    kept = cands_scored[(cands_scored["score"] >= threshold) & guard_ok]
    grouped = kept.groupby("source1_entity_id")["candidate_id"].apply(lambda ids: ",".join(sorted(set(ids))))
    out = pd.DataFrame({"source1_entity_id": list(all_s1_ids)})
    out = out.merge(grouped.rename("matched_entity_ids"), on="source1_entity_id", how="left")
    out["matched_entity_ids"] = out["matched_entity_ids"].fillna("")
    return out


def tune_threshold(cands_scored: pd.DataFrame, gt_val: pd.DataFrame, all_s1_ids, thresholds=np.arange(0.30, 0.96, 0.02)):
    best_t, best_f05 = 0.5, -1.0
    for t in thresholds:
        pred = predictions_to_df(cands_scored, t, all_s1_ids)
        result = score_submission(pred, gt_val)
        if result["macro_f0_5"] > best_f05:
            best_f05, best_t = result["macro_f0_5"], t
    print(f"[threshold] best={best_t:.2f}  macro F0.5={best_f05:.4f}")
    return best_t, best_f05


def write_tsv(df: pd.DataFrame, path: Path, id_col: str, list_col: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df[[id_col, list_col]].to_csv(path, sep="\t", index=False)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--n-jobs", type=int, default=1, help="parallelize featurization across N processes")
    args = ap.parse_args()
    root, out = Path(args.data_dir), Path(args.out_dir)

    load = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    train_s1 = prepare(load(root / "train" / "train_source1.tsv"))
    train_s2 = prepare(load(root / "train" / "train_source2.tsv"))
    train_s3 = prepare(load(root / "train" / "train_source3.tsv"))
    gt = load(root / "train" / "train_ground_truth.tsv")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=0)
    tr_idx, va_idx = next(gss.split(np.arange(len(train_s1)), groups=train_s1["entity_id"]))
    s1_tr, s1_va = train_s1.iloc[tr_idx], train_s1.iloc[va_idx]
    gt_tr = gt[gt["source1_entity_id"].isin(s1_tr["entity_id"])]
    gt_va = gt[gt["source1_entity_id"].isin(s1_va["entity_id"])]

    print("=== training candidates ===")
    cand_tr = build_candidates_all(s1_tr, train_s2, train_s3, top_k_per_entity=args.top_k)
    other_all = pd.concat([train_s2, train_s3], ignore_index=True)
    feat_tr = featurize_candidates(cand_tr, s1_tr, other_all, n_jobs=args.n_jobs)
    labeled_tr = build_labeled_set(feat_tr, gt_tr)
    model = train_classifier(labeled_tr)

    print("=== validation candidates ===")
    cand_va = build_candidates_all(s1_va, train_s2, train_s3, top_k_per_entity=args.top_k)
    feat_va = featurize_candidates(cand_va, s1_va, other_all, n_jobs=args.n_jobs)
    feat_va["score"] = model.predict_proba(feat_va[FEATURE_NAMES])[:, 1]
    best_t, best_f05 = tune_threshold(feat_va, gt_va, s1_va["entity_id"])

    if HAVE_LGBM:
        imp = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
        print("\n[feature importance]\n", imp.to_string())

    print("\n=== refitting on full train, scoring test set ===")
    test_s1 = prepare(load(root / "test" / "test_source1.tsv"))
    test_s2 = prepare(load(root / "test" / "test_source2.tsv"))
    test_s3 = prepare(load(root / "test" / "test_source3.tsv"))

    cand_full = build_candidates_all(train_s1, train_s2, train_s3, top_k_per_entity=args.top_k)
    feat_full = featurize_candidates(cand_full, train_s1, other_all, n_jobs=args.n_jobs)
    labeled_full = build_labeled_set(feat_full, gt)
    final_model = train_classifier(labeled_full)

    test_other = pd.concat([test_s2, test_s3], ignore_index=True)
    cand_test = build_candidates_all(test_s1, test_s2, test_s3, top_k_per_entity=args.top_k)
    feat_test = featurize_candidates(cand_test, test_s1, test_other, n_jobs=args.n_jobs)
    feat_test["score"] = final_model.predict_proba(feat_test[FEATURE_NAMES])[:, 1]

    candidate_out = feat_test.groupby("source1_entity_id")["candidate_id"].apply(lambda ids: ",".join(sorted(set(ids))))
    candidate_out = pd.DataFrame({"source1_entity_id": test_s1["entity_id"]}).merge(
        candidate_out.rename("candidate_entity_ids"), on="source1_entity_id", how="left"
    )
    candidate_out["candidate_entity_ids"] = candidate_out["candidate_entity_ids"].fillna("")
    write_tsv(candidate_out, out / "candidate_pairs.tsv", "source1_entity_id", "candidate_entity_ids")

    matching_out = predictions_to_df(feat_test, best_t, test_s1["entity_id"])
    write_tsv(matching_out, out / "matching_results.tsv", "source1_entity_id", "matched_entity_ids")

    print(f"\nwrote {out/'candidate_pairs.tsv'} and {out/'matching_results.tsv'}")
    print(f"validation macro F0.5 at chosen threshold: {best_f05:.4f}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Scale notes (measured on a synthetic ~30k S1 / 64k S2 / 64k S3 multi-country,
# multi-script dataset, on a constrained 1-vCPU / 4GB sandbox -- re-measure on
# your actual hardware and real data, but use this as a starting point):
#
#   prepare()              : ~120k rows/sec
#   build_candidates_all() : ~1.7M candidate pairs from 30k x 128k in ~4s
#   featurize_candidates() : ~35k pairs/sec single-core (the dominant cost --
#                             use --n-jobs on a real multi-core machine; the
#                             parallel path is verified bit-identical to serial)
#   train_classifier()     : ~36s for 1.8M rows x 18 features (LightGBM)
#
# Two things this testing surfaced that matter more than the raw numbers:
#
#  1. top_k_per_entity has a real recall/speed trade-off, and the right value
#     is data-dependent -- on the synthetic set, recall was 67% at top_k=50
#     and 87% at top_k=150 before plateauing. Before committing to a value,
#     run build_candidates_all() at a few top_k settings against your real
#     train + ground truth and plot recall vs. candidate-set size, the same
#     way the block above does it. Whatever plateau you find IS your
#     blocking recall ceiling -- nothing downstream can recover it.
#
#  2. Any block key that can legitimately cover a large population --
#     a generic name prefix, a coarse postal/PIN code -- needs a hard
#     max_block_size cap, or a single common key silently turns an O(n) hash
#     join into an O(n*m) blow-up. This isn't a synthetic-data quirk: Indian
#     PIN codes and generic business-name patterns make this a live risk in
#     the real data too, which is exactly what postal_block/prefix_block's
#     max_block_size guards against here.
# --------------------------------------------------------------------------
