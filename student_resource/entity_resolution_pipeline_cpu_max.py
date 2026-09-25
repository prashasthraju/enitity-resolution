#!/usr/bin/env python3
"""
Business Entity Resolution Challenge 2026 - end-to-end baseline.

Pipeline:
    1. ingest TSVs into SQLite
    2. country-aware multilingual preprocessing
    3. multi-pass blocking (union of blocks)
    4. hard-negative / positive sampling from blocked pairs
    5. LightGBM pair classifier
    6. validation threshold tuning for macro F0.5
    7. test inference
    8. write matching_results.tsv + candidate_pairs.tsv
    9. optionally invoke the supplied submission validator

Expected directories:
    dataset/train/{train_source1.tsv,train_source2.tsv,train_source3.tsv,train_ground_truth.tsv}
    dataset/test/{test_source1.tsv,test_source2.tsv,test_source3.tsv}

Usage:
    python entity_resolution_pipeline.py \
        --train-dir dataset/train \
        --test-dir dataset/test \
        --output-dir output \
        --work-dir work

The code intentionally does NOT overwrite original input fields; normalized fields
live in the SQLite working database. Country is treated as a generic string label,
while India/US/France get extra address canonicalization rules when present.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode


CPU_WORKERS = max(1, min(32, os.cpu_count() or 1))

FEATURES = [
    "name_exact",
    "name_core_exact",
    "name_sorted_exact",
    "name_ratio",
    "name_token_ratio",
    "name_first_exact",
    "name_len_sim",
    "address_exact",
    "address_sorted_exact",
    "address_ratio",
    "address_token_ratio",
    "address_len_sim",
    "postal_equal",
    "house_equal",
    "address_mid_equal",
    "name_nonempty_both",
    "address_nonempty_both",
    "block_count",
    "block_name_exact",
    "block_name_core",
    "block_name_sorted",
    "block_name_prefix",
    "block_address_exact",
    "block_address_sorted",
    "block_postal_name",
    "block_house_name",
    "block_house_mid",
    "block_postal_mid",
    "block_house_postal",
    "match_is_s2",
]

BLOCK_BITS = {
    "name_exact": 1,
    "name_core": 2,
    "name_sorted": 4,
    "name_prefix": 8,
    "address_exact": 16,
    "address_sorted": 32,
    "postal_name": 64,
    "house_name": 128,
    "house_mid": 256,
    "postal_mid": 512,
    "house_postal": 1024,
}

NULL_STRINGS = {"", "null", "none", "nan", "na", "n/a", "<na>"}

NAME_SUFFIX_PATTERN = re.compile(
    r"\b(?:incorporated|corporation|company|limited|private|proprietary|"
    r"llp|llc|ltd|inc|corp|co|pvt|plc|gmbh|sarl|sas|sa)\b",
    flags=re.IGNORECASE,
)

GENERIC_ADDR_REPLACEMENTS = {
    "street": "st",
    "st": "st",
    "road": "rd",
    "rd": "rd",
    "avenue": "ave",
    "av": "ave",
    "ave": "ave",
    "boulevard": "blvd",
    "blvd": "blvd",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "highway": "hwy",
    "hwy": "hwy",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "place": "pl",
    "pl": "pl",
    "suite": "ste",
    "ste": "ste",
    "apartment": "apt",
    "apt": "apt",
    "number": "no",
    "no": "no",
    "near": "nr",
    "nr": "nr",
}

COUNTRY_ADDR_REPLACEMENTS = {
    "India": {
        "marg": "rd",
        "marga": "rd",
        "nagar": "nagar",
        "colony": "colony",
    },
    "US": {
        "saint": "st",
        "mount": "mt",
    },
    "France": {
        "rue": "rue",
        "avenue": "ave",
        "boulevard": "blvd",
        "chemin": "chemin",
        "impasse": "impasse",
    },
}

US_STATE_MAP = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}

POSTAL_PATTERNS = {
    "India": re.compile(r"\b[1-9][0-9]{5}\b"),
    "US": re.compile(r"\b[0-9]{5}(?:-[0-9]{4})?\b"),
    "France": re.compile(r"\b[0-9]{5}\b"),
}
HOUSE_RE = re.compile(r"\b[0-9]{1,6}[A-Za-z]?(?:[-/][0-9A-Za-z]+)?\b")


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in NULL_STRINGS:
        return ""
    return unicodedata.normalize("NFKC", s)


def to_ascii(s: str) -> str:
    if not s:
        return ""
    return unidecode(s) if not s.isascii() else s


def compact_words(s: str) -> str:
    s = to_ascii(clean_text(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_name(s: str) -> str:
    return compact_words(s)


def normalize_name_core(s: str) -> str:
    s = normalize_name(s)
    if not s:
        return ""
    s = NAME_SUFFIX_PATTERN.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def sorted_tokens(s: str) -> str:
    if not s:
        return ""
    toks = sorted(set(t for t in s.split() if t))
    return " ".join(toks)


def first_token(s: str) -> str:
    return s.split()[0] if s else ""


def first_two_tokens(s: str) -> str:
    toks = sorted(set(s.split()))
    return " ".join(toks[:2])


def normalize_address_with_commas(raw: str, country: str) -> Tuple[str, str, str, str]:
    """Return normalized address, sorted-token key, second-last segment, last segment."""
    s = clean_text(raw)
    if not s:
        return "", "", "", ""
    s = to_ascii(s).lower()
    s = s.replace(";", ",").replace("|", ",")
    # Keep commas long enough to recover rough address segments.
    s = re.sub(r"[^a-z0-9,/#-]+", " ", s)
    s = re.sub(r"\s*,\s*", ",", s)
    segments = [re.sub(r"\s+", " ", x).strip(" -") for x in s.split(",")]
    segments = [x for x in segments if x]

    replacements = dict(GENERIC_ADDR_REPLACEMENTS)
    replacements.update(COUNTRY_ADDR_REPLACEMENTS.get(country, {}))
    for k, v in replacements.items():
        segments = [re.sub(rf"\b{re.escape(k)}\b", v, x) for x in segments]

    if country == "US":
        # Convert full state names to two-letter abbreviations before compacting.
        for full, abbr in sorted(US_STATE_MAP.items(), key=lambda kv: -len(kv[0])):
            segments = [re.sub(rf"\b{re.escape(full)}\b", abbr, x) for x in segments]

    comma_norm = ",".join(segments)
    flat = re.sub(r"[^a-z0-9]+", " ", comma_norm).strip()
    flat = re.sub(r"\s+", " ", flat)
    sorted_key = sorted_tokens(flat)
    mid = segments[-2] if len(segments) >= 2 else ""
    last = segments[-1] if segments else ""
    mid = compact_words(mid)
    last = compact_words(last)
    return flat, sorted_key, mid, last


def extract_postal(address: str, country: str) -> str:
    if not address:
        return ""
    pat = POSTAL_PATTERNS.get(country, re.compile(r"\b[0-9]{5,6}\b"))
    m = pat.search(address)
    return m.group(0).replace("-", "") if m else ""


def extract_house(address: str) -> str:
    if not address:
        return ""
    m = HOUSE_RE.search(address)
    return m.group(0).lower() if m else ""


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df[["entity_id", "business_name", "business_address", "country"]].copy()
    out["entity_id"] = out["entity_id"].astype(str)
    out["country"] = out["country"].map(clean_text)
    out["business_name"] = out["business_name"].map(clean_text)
    out["business_address"] = out["business_address"].map(clean_text)

    # Names.
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["name_core"] = out["name_norm"].map(normalize_name_core)
    out["name_sorted"] = out["name_norm"].map(sorted_tokens)
    out["name_first"] = out["name_norm"].map(first_token)
    out["name_prefix"] = out["name_norm"].map(lambda x: x.replace(" ", "")[:8] if len(x.replace(" ", "")) >= 8 else "")
    out["name_sig2"] = out["name_norm"].map(first_two_tokens)

    # Addresses.
    addr_parts = [
        normalize_address_with_commas(raw, country)
        for raw, country in zip(out["business_address"].to_numpy(), out["country"].to_numpy())
    ]
    out["address_norm"] = [x[0] for x in addr_parts]
    out["address_sorted"] = [x[1] for x in addr_parts]
    out["address_mid"] = [x[2] for x in addr_parts]
    out["address_last"] = [x[3] for x in addr_parts]
    out["postal"] = [extract_postal(x[0], c) for x, c in zip(addr_parts, out["country"])]
    out["house"] = [extract_house(x[0]) for x in addr_parts]

    # Candidate block keys are kept explicitly for transparent debugging.
    out["key_name"] = out["country"] + "|" + out["name_norm"]
    out["key_name_core"] = out["country"] + "|" + out["name_core"]
    out["key_name_sorted"] = out["country"] + "|" + out["name_sorted"]
    out["key_name_prefix"] = out["country"] + "|" + out["name_prefix"]
    out["key_address"] = out["country"] + "|" + out["address_norm"]
    out["key_address_sorted"] = out["country"] + "|" + out["address_sorted"]
    out["key_postal_name"] = out["country"] + "|" + out["postal"] + "|" + out["name_first"]
    out["key_house_name"] = out["country"] + "|" + out["house"] + "|" + out["name_first"]
    out["key_house_mid"] = out["country"] + "|" + out["house"] + "|" + out["address_mid"]
    out["key_postal_mid"] = out["country"] + "|" + out["postal"] + "|" + out["address_mid"]
    out["key_house_postal"] = out["country"] + "|" + out["house"] + "|" + out["postal"]

    # Empty block keys are disabled by making them empty.
    key_cols = [c for c in out.columns if c.startswith("key_")]
    for c in key_cols:
        out.loc[out[c].str.endswith("|"), c] = ""

    return out


DB_FIELDS = [
    "entity_id", "country",
    "name_norm", "name_core", "name_sorted", "name_first", "name_prefix", "name_sig2",
    "address_norm", "address_sorted", "address_mid", "address_last", "postal", "house",
    "key_name", "key_name_core", "key_name_sorted", "key_name_prefix",
    "key_address", "key_address_sorted", "key_postal_name", "key_house_name",
    "key_house_mid", "key_postal_mid", "key_house_postal",
]


def open_db(path: Path, cache_gb: int = 12, mmap_gb: int = 32) -> sqlite3.Connection:
    """Open a disposable working DB tuned for this machine.

    The DB is rebuilt from source data, so we prioritize throughput over crash
    durability. A large page cache + mmap keeps repeated blocking joins fast.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA locking_mode=EXCLUSIVE")
    con.execute("PRAGMA temp_store=FILE")
    con.execute(f"PRAGMA cache_size={-int(cache_gb * 1_000_000 / 1)}")
    con.execute(f"PRAGMA mmap_size={int(mmap_gb) * 1024 * 1024 * 1024}")
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute("PRAGMA automatic_index=ON")
    return con


def init_record_db(con: sqlite3.Connection) -> None:
    cols = [f"{c} TEXT" for c in DB_FIELDS]
    con.execute(f"CREATE TABLE records ({', '.join(cols)}, source TEXT NOT NULL, PRIMARY KEY(entity_id, source))")
    con.execute("CREATE TABLE candidates (s1_id TEXT NOT NULL, match_id TEXT NOT NULL, block_mask INTEGER NOT NULL, PRIMARY KEY(s1_id, match_id))")
    con.execute("CREATE TABLE ground_truth (s1_id TEXT NOT NULL, match_id TEXT NOT NULL, PRIMARY KEY(s1_id, match_id))")
    con.execute("CREATE TABLE s1_stats (s1_id TEXT PRIMARY KEY, match_count INTEGER NOT NULL, split TEXT NOT NULL)")
    con.commit()


def create_record_indexes(con: sqlite3.Connection) -> None:
    for key in [
        "key_name", "key_name_core", "key_name_sorted", "key_name_prefix",
        "key_address", "key_address_sorted", "key_postal_name", "key_house_name",
        "key_house_mid", "key_postal_mid", "key_house_postal",
    ]:
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{key} ON records(source, {key})")
    con.execute("CREATE INDEX IF NOT EXISTS idx_records_id ON records(entity_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_candidates_s1 ON candidates(s1_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_candidates_mid ON candidates(match_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_gt_s1 ON ground_truth(s1_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_gt_mid ON ground_truth(match_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_s1_stats_split ON s1_stats(split,s1_id)")
    con.commit()


def insert_records(con: sqlite3.Connection, df: pd.DataFrame, source: str, chunk_name: str = "") -> None:
    """Bulk insert without materializing a huge list of Python dictionaries."""
    cols = DB_FIELDS
    sql = f"INSERT INTO records ({','.join(cols)},source) VALUES ({','.join(['?'] * (len(cols) + 1))})"
    rows = (tuple(row) + (source,) for row in df[cols].itertuples(index=False, name=None))
    con.executemany(sql, rows)
    con.commit()
    log(f"Inserted {len(df):,} {source} rows {chunk_name}".strip())


def load_sources_into_db(con: sqlite3.Connection, data_dir: Path, prefix: str, chunk_size: int = 250_000) -> pd.DataFrame:
    # S1 is small enough to keep its IDs; S2/S3 are streamed to avoid holding all
    # 10M+ raw records in RAM at once.
    s1_path = data_dir / f"{prefix}_source1.tsv"
    s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s1_p = preprocess_df(s1)
    insert_records(con, s1_p, "S1")
    s1_ids = s1[["entity_id"]].copy()
    del s1, s1_p

    for source, num in [("S2", 2), ("S3", 3)]:
        path = data_dir / f"{prefix}_source{num}.tsv"
        total = 0
        for i, df in enumerate(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size), 1):
            p = preprocess_df(df)
            insert_records(con, p, source, chunk_name=f"chunk {i}")
            total += len(p)
            del df, p
        log(f"Finished {source}: {total:,} rows")

    create_record_indexes(con)
    return s1_ids


def load_ground_truth(con: sqlite3.Connection, gt_path: Path, s1_ids: Iterable[str]) -> None:
    log("Loading ground truth into SQLite...")
    cur = con.cursor()
    insert_sql = "INSERT OR IGNORE INTO ground_truth(s1_id,match_id) VALUES (?,?)"
    stats: Dict[str, int] = {}

    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, chunksize=100_000):
        rows = []
        for s1_id, matched in chunk.itertuples(index=False, name=None):
            s1_id = str(s1_id).strip()
            matched = clean_text(matched)
            if not matched:
                stats[s1_id] = 0
                continue
            mids = [x.strip() for x in matched.split(",") if x.strip()]
            stats[s1_id] = len(set(mids))
            rows.extend((s1_id, mid) for mid in set(mids))
        if rows:
            cur.executemany(insert_sql, rows)
            con.commit()
        log(f"GT chunk processed; cumulative ground-truth pairs in DB={cur.execute('SELECT COUNT(*) FROM ground_truth').fetchone()[0]:,}")

    # Some S1 IDs can have duplicate GT rows in pathological input. Recompute exact match counts from DB.
    counts = dict(con.execute("SELECT s1_id, COUNT(*) FROM ground_truth GROUP BY s1_id"))
    split_rows = []
    for s1_id in s1_ids:
        n = int(counts.get(s1_id, 0))
        h = int(hashlib.md5(str(s1_id).encode()).hexdigest()[:8], 16) % 100
        split = "val" if h < 20 else "train"
        split_rows.append((str(s1_id), n, split))
    con.executemany("INSERT OR REPLACE INTO s1_stats(s1_id,match_count,split) VALUES (?,?,?)", split_rows)
    con.commit()
    create_record_indexes(con)
    log(f"Ground truth loaded: {sum(counts.values()):,} positive pairs; singleton S1={sum(1 for _, n, _ in split_rows if n == 0):,}")


def add_block(con: sqlite3.Connection, source: str, s2_key: str, bit: int, where: str = "") -> None:
    sql = f"""
    INSERT INTO candidates(s1_id, match_id, block_mask)
    SELECT a.entity_id, b.entity_id, ?
    FROM records a
    JOIN records b
      ON a.source='S1' AND b.source=?
     AND a.{s2_key}=b.{s2_key}
     AND a.{s2_key}<>''
    {where}
    ON CONFLICT(s1_id,match_id) DO UPDATE SET block_mask = candidates.block_mask | excluded.block_mask
    """
    con.execute(sql, (bit, source))
    con.commit()


def generate_candidates(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM candidates")
    con.commit()
    blocks = [
        ("key_name", "name_exact"),
        ("key_name_core", "name_core"),
        ("key_name_sorted", "name_sorted"),
        ("key_name_prefix", "name_prefix"),
        ("key_address", "address_exact"),
        ("key_address_sorted", "address_sorted"),
        ("key_postal_name", "postal_name"),
        ("key_house_name", "house_name"),
        ("key_house_mid", "house_mid"),
        ("key_postal_mid", "postal_mid"),
        ("key_house_postal", "house_postal"),
    ]
    for key, block_name in blocks:
        bit = BLOCK_BITS[block_name]
        where = ""
        if key == "key_name_prefix":
            where = "WHERE length(a.name_prefix)>=8 AND length(b.name_prefix)>=8"
        log(f"Blocking on {block_name} ...")
        t = time.time()
        for source in ("S2", "S3"):
            add_block(con, source, key, bit, where)
        n = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        log(f"  candidates after {block_name}: {n:,} ({time.time()-t:.1f}s)")

    log("Computing candidate statistics...")
    stats = con.execute("""
        SELECT COUNT(*) AS pairs,
               AVG(cnt) AS avg_candidates,
               MAX(cnt) AS max_candidates
        FROM (SELECT s1_id, COUNT(*) cnt FROM candidates GROUP BY s1_id)
    """).fetchone()
    with_match = con.execute("""
        SELECT COUNT(*)
        FROM s1_stats s
        WHERE s.match_count>0
          AND EXISTS (SELECT 1 FROM candidates c WHERE c.s1_id=s.s1_id
                      AND EXISTS (SELECT 1 FROM ground_truth g WHERE g.s1_id=c.s1_id AND g.match_id=c.match_id))
    """).fetchone()[0]
    gt_pairs = con.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    covered_pairs = con.execute("""
        SELECT COUNT(*) FROM ground_truth g
        WHERE EXISTS (SELECT 1 FROM candidates c WHERE c.s1_id=g.s1_id AND c.match_id=g.match_id)
    """).fetchone()[0]
    s1_total = con.execute("SELECT COUNT(*) FROM s1_stats WHERE match_count>0").fetchone()[0]
    log(f"Candidate pairs={stats[0]:,}; avg/S1={stats[1]:.2f}; max/S1={stats[2]:,}")
    log(f"Blocking pair recall={covered_pairs/max(gt_pairs,1):.4%}; matched-S1 coverage={with_match/max(s1_total,1):.4%}")


def materialize_train_pairs(con: sqlite3.Connection, max_pos: int, max_neg: int, singleton_neg: int) -> None:
    con.execute("DROP TABLE IF EXISTS train_pairs")
    con.execute("CREATE TABLE train_pairs (s1_id TEXT, match_id TEXT, block_mask INTEGER, label INTEGER)")
    con.execute("CREATE INDEX idx_train_pairs_s1 ON train_pairs(s1_id)")

    pos_sql = """
    INSERT INTO train_pairs
    SELECT c.s1_id, c.match_id, c.block_mask, 1
    FROM candidates c
    JOIN ground_truth g ON g.s1_id=c.s1_id AND g.match_id=c.match_id
    JOIN s1_stats s ON s.s1_id=c.s1_id
    WHERE s.split='train'
      AND abs(random()%100) < 50
    LIMIT ?
    """
    neg_sql = """
    INSERT INTO train_pairs
    SELECT c.s1_id, c.match_id, c.block_mask, 0
    FROM candidates c
    JOIN s1_stats s ON s.s1_id=c.s1_id
    WHERE s.split='train'
      AND abs(random()%100) < 15
      AND NOT EXISTS (SELECT 1 FROM ground_truth g WHERE g.s1_id=c.s1_id AND g.match_id=c.match_id)
    LIMIT ?
    """
    singleton_sql = """
    INSERT INTO train_pairs
    SELECT c.s1_id, c.match_id, c.block_mask, 0
    FROM candidates c
    JOIN s1_stats s ON s.s1_id=c.s1_id AND s.match_count=0
    WHERE abs(random()%100) < 40
    LIMIT ?
    """
    con.execute(pos_sql, (max_pos,))
    con.execute(neg_sql, (max_neg,))
    con.execute(singleton_sql, (singleton_neg,))
    con.commit()

    dup = con.execute("""
      SELECT s1_id, match_id, COUNT(*) FROM train_pairs
      GROUP BY s1_id, match_id HAVING COUNT(*)>1 LIMIT 1
    """).fetchone()
    if dup:
        con.execute("""
          DELETE FROM train_pairs
          WHERE rowid NOT IN (SELECT MIN(rowid) FROM train_pairs GROUP BY s1_id,match_id)
        """)
        con.commit()
    counts = dict(con.execute("SELECT label, COUNT(*) FROM train_pairs GROUP BY label"))
    log(f"Training sample: positives={counts.get(1,0):,}, negatives={counts.get(0,0):,}")


def decode_mask(mask: int) -> Dict[str, int]:
    return {name: int(mask & bit != 0) for name, bit in BLOCK_BITS.items()}


def feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    # The SQL query already brings only normalized strings, so this function is deliberately compact.
    out = pd.DataFrame(index=rows.index)

    n1 = rows["s1_name"].fillna("").astype(str).to_numpy()
    n2 = rows["m_name"].fillna("").astype(str).to_numpy()
    nc1 = rows["s1_name_core"].fillna("").astype(str).to_numpy()
    nc2 = rows["m_name_core"].fillna("").astype(str).to_numpy()
    ns1 = rows["s1_name_sorted"].fillna("").astype(str).to_numpy()
    ns2 = rows["m_name_sorted"].fillna("").astype(str).to_numpy()
    a1 = rows["s1_address"].fillna("").astype(str).to_numpy()
    a2 = rows["m_address"].fillna("").astype(str).to_numpy()
    as1 = rows["s1_address_sorted"].fillna("").astype(str).to_numpy()
    as2 = rows["m_address_sorted"].fillna("").astype(str).to_numpy()

    def ratio_arr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # cpdist is specifically for aligned pairs and can parallelize the C-API scorer.
        if len(a) == 0:
            return np.empty(0, dtype=np.float32)
        return (
            process.cpdist(
                a.tolist(), b.tolist(), scorer=fuzz.ratio,
                workers=CPU_WORKERS, dtype=np.float32, score_multiplier=0.01
            )
        ).reshape(-1)

    def token_ratio_arr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(a) == 0:
            return np.empty(0, dtype=np.float32)
        return (
            process.cpdist(
                a.tolist(), b.tolist(), scorer=fuzz.token_set_ratio,
                workers=CPU_WORKERS, dtype=np.float32, score_multiplier=0.01
            )
        ).reshape(-1)

    def len_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        outv = np.zeros(len(a), dtype=np.float32)
        for i, (x, y) in enumerate(zip(a, b)):
            m = max(len(x), len(y), 1)
            outv[i] = 1.0 - abs(len(x)-len(y))/m
        return outv

    out["name_exact"] = (n1 == n2).astype(np.float32)
    out["name_core_exact"] = ((nc1 != "") & (nc1 == nc2)).astype(np.float32)
    out["name_sorted_exact"] = ((ns1 != "") & (ns1 == ns2)).astype(np.float32)
    out["name_ratio"] = ratio_arr(n1, n2)
    out["name_token_ratio"] = token_ratio_arr(n1, n2)
    out["name_first_exact"] = (rows["s1_name_first"].fillna("").to_numpy() == rows["m_name_first"].fillna("").to_numpy()).astype(np.float32)
    out["name_len_sim"] = len_sim(n1, n2)

    out["address_exact"] = ((a1 != "") & (a1 == a2)).astype(np.float32)
    out["address_sorted_exact"] = ((as1 != "") & (as1 == as2)).astype(np.float32)
    out["address_ratio"] = ratio_arr(a1, a2)
    out["address_token_ratio"] = token_ratio_arr(a1, a2)
    out["address_len_sim"] = len_sim(a1, a2)

    out["postal_equal"] = ((rows["s1_postal"].fillna("").to_numpy() != "") & (rows["s1_postal"].fillna("").to_numpy() == rows["m_postal"].fillna("").to_numpy())).astype(np.float32)
    s1_house = rows["s1_house"].fillna("").to_numpy()
    m_house = rows["m_house"].fillna("").to_numpy()
    out["house_equal"] = ((s1_house != "") & (s1_house == m_house)).astype(np.float32)
    s1_mid = rows["s1_mid"].fillna("").to_numpy()
    m_mid = rows["m_mid"].fillna("").to_numpy()
    out["address_mid_equal"] = ((s1_mid != "") & (s1_mid == m_mid)).astype(np.float32)
    out["name_nonempty_both"] = ((n1 != "") & (n2 != "")).astype(np.float32)
    out["address_nonempty_both"] = ((a1 != "") & (a2 != "")).astype(np.float32)

    masks = rows["block_mask"].fillna(0).astype(np.int64).to_numpy()
    out["block_count"] = np.fromiter((int(x).bit_count() for x in masks), dtype=np.float32, count=len(masks))
    for block_name, bit in BLOCK_BITS.items():
        out["block_" + block_name.replace("_exact", "_exact")] = ((masks & bit) != 0).astype(np.float32)

    out["match_is_s2"] = rows["match_id"].astype(str).str.startswith("S2-").astype(np.float32).to_numpy()
    return out[FEATURES]


def pair_query(table: str, extra_where: str = "", limit: int | None = None, offset_rowid: int = 0) -> str:
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
    return f"""
    SELECT c.rowid AS c_rowid, c.s1_id, c.match_id, c.block_mask,
           a.name_norm AS s1_name, b.name_norm AS m_name,
           a.name_core AS s1_name_core, b.name_core AS m_name_core,
           a.name_sorted AS s1_name_sorted, b.name_sorted AS m_name_sorted,
           a.name_first AS s1_name_first, b.name_first AS m_name_first,
           a.address_norm AS s1_address, b.address_norm AS m_address,
           a.address_sorted AS s1_address_sorted, b.address_sorted AS m_address_sorted,
           a.postal AS s1_postal, b.postal AS m_postal,
           a.house AS s1_house, b.house AS m_house,
           a.address_mid AS s1_mid, b.address_mid AS m_mid,
           a.source AS s1_source, b.source AS m_source
    FROM {table} c
    JOIN records a ON a.entity_id=c.s1_id AND a.source='S1'
    JOIN records b ON b.entity_id=c.match_id
    WHERE c.rowid > {int(offset_rowid)} {extra_where}
    ORDER BY c.rowid
    {limit_sql}
    """


def stream_pair_features(con: sqlite3.Connection, table: str, where: str = "", chunk_size: int = 100_000, with_labels: bool = False) -> Iterator[pd.DataFrame]:
    last_rowid = 0
    while True:
        label_sql = ", c.label" if table == "train_pairs" else ""
        gt_join = ""
        gt_sql = ""
        if with_labels and table != "train_pairs":
            gt_join = "LEFT JOIN ground_truth g ON g.s1_id=c.s1_id AND g.match_id=c.match_id"
            gt_sql = ", CASE WHEN g.s1_id IS NULL THEN 0 ELSE 1 END AS label"
        q = f"""
        SELECT c.rowid AS c_rowid, c.s1_id, c.match_id, c.block_mask{label_sql},
               a.name_norm AS s1_name, b.name_norm AS m_name,
               a.name_core AS s1_name_core, b.name_core AS m_name_core,
               a.name_sorted AS s1_name_sorted, b.name_sorted AS m_name_sorted,
               a.name_first AS s1_name_first, b.name_first AS m_name_first,
               a.address_norm AS s1_address, b.address_norm AS m_address,
               a.address_sorted AS s1_address_sorted, b.address_sorted AS m_address_sorted,
               a.postal AS s1_postal, b.postal AS m_postal,
               a.house AS s1_house, b.house AS m_house,
               a.address_mid AS s1_mid, b.address_mid AS m_mid,
               a.source AS s1_source, b.source AS m_source{gt_sql}
        FROM {table} c
        JOIN records a ON a.entity_id=c.s1_id AND a.source='S1'
        JOIN records b ON b.entity_id=c.match_id
        {gt_join}
        WHERE c.rowid > {int(last_rowid)} {where}
        ORDER BY c.rowid
        LIMIT {chunk_size}
        """
        df = pd.read_sql_query(q, con)
        if df.empty:
            break
        yield df
        last_rowid = int(df["c_rowid"].iloc[-1])


def collect_training_arrays(con: sqlite3.Connection, max_rows: int = 14_000_000) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    seen = 0
    for df in stream_pair_features(con, "train_pairs", chunk_size=150_000, with_labels=False):
        X = feature_frame(df)
        y = df["label"].to_numpy(np.int8)
        take = min(len(df), max_rows - seen)
        xs.append(X.to_numpy(np.float32)[:take])
        ys.append(y[:take])
        seen += take
        if seen >= max_rows:
            break
    X = np.vstack(xs)
    y = np.concatenate(ys)
    log(f"Feature matrix: {X.shape}, positives={int(y.sum()):,}, negatives={int((y==0).sum()):,}")
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int = 1200) -> lgb.LGBMClassifier:
    """CPU-only high-throughput LightGBM configuration."""
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.90,
        colsample_bytree=0.95,
        reg_alpha=0.20,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=CPU_WORKERS,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def validation_arrays(con: sqlite3.Connection, model: lgb.LGBMClassifier, chunk_size: int = 150_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    con.execute("DROP TABLE IF EXISTS val_s1_map")
    val_ids = [r[0] for r in con.execute("SELECT s1_id FROM s1_stats WHERE split='val' ORDER BY s1_id")]
    con.execute("CREATE TABLE val_s1_map(s1_id TEXT PRIMARY KEY, idx INTEGER NOT NULL)")
    con.executemany("INSERT INTO val_s1_map(s1_id,idx) VALUES (?,?)", [(x,i) for i,x in enumerate(val_ids)])
    con.commit()
    true_counts = np.array([r[0] for r in con.execute("SELECT match_count FROM s1_stats WHERE split='val' ORDER BY s1_id")], dtype=np.int32)

    idxs, scores, labels = [], [], []
    last = 0
    while True:
        q = f"""
        SELECT c.rowid AS c_rowid, v.idx AS s1_idx, c.s1_id, c.match_id, c.block_mask,
               a.name_norm AS s1_name, b.name_norm AS m_name,
               a.name_core AS s1_name_core, b.name_core AS m_name_core,
               a.name_sorted AS s1_name_sorted, b.name_sorted AS m_name_sorted,
               a.name_first AS s1_name_first, b.name_first AS m_name_first,
               a.address_norm AS s1_address, b.address_norm AS m_address,
               a.address_sorted AS s1_address_sorted, b.address_sorted AS m_address_sorted,
               a.postal AS s1_postal, b.postal AS m_postal,
               a.house AS s1_house, b.house AS m_house,
               a.address_mid AS s1_mid, b.address_mid AS m_mid,
               b.source AS m_source,
               CASE WHEN g.s1_id IS NULL THEN 0 ELSE 1 END AS label
        FROM candidates c
        JOIN val_s1_map v ON v.s1_id=c.s1_id
        JOIN records a ON a.entity_id=c.s1_id AND a.source='S1'
        JOIN records b ON b.entity_id=c.match_id
        LEFT JOIN ground_truth g ON g.s1_id=c.s1_id AND g.match_id=c.match_id
        WHERE c.rowid>{last}
        ORDER BY c.rowid
        LIMIT {chunk_size}
        """
        df = pd.read_sql_query(q, con)
        if df.empty:
            break
        df["label"] = df["label"].astype(np.int8)
        X = feature_frame(df)
        s = model.predict_proba(X)[:,1].astype(np.float32)
        idxs.append(df["s1_idx"].to_numpy(np.int32))
        scores.append(s)
        labels.append(df["label"].to_numpy(np.int8))
        last = int(df["c_rowid"].iloc[-1])
    return true_counts, np.concatenate(idxs), np.concatenate(scores), np.concatenate(labels)


def f05_arrays(true_counts: np.ndarray, s1_idx: np.ndarray, scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    pred = scores >= threshold
    tp = np.bincount(s1_idx, weights=(pred & (labels == 1)).astype(np.float32), minlength=len(true_counts))
    pred_n = np.bincount(s1_idx, weights=pred.astype(np.float32), minlength=len(true_counts))
    precision = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n>0)
    recall = np.divide(tp, true_counts, out=np.zeros_like(tp), where=true_counts>0)
    f = np.divide(1.25*precision*recall, 0.25*precision + recall, out=np.zeros_like(tp), where=(0.25*precision+recall)>0)
    f[(true_counts==0) & (pred_n==0)] = 1.0
    return float(f.mean())


def tune_threshold(true_counts, s1_idx, scores, labels) -> Tuple[float, float]:
    coarse = np.linspace(0.10, 0.99, 46)
    vals = [(float(t), f05_arrays(true_counts, s1_idx, scores, labels, float(t))) for t in coarse]
    best_t, best_f = max(vals, key=lambda x: x[1])
    fine_lo = max(0.01, best_t - 0.04)
    fine_hi = min(0.999, best_t + 0.04)
    fine = np.linspace(fine_lo, fine_hi, 41)
    vals2 = [(float(t), f05_arrays(true_counts, s1_idx, scores, labels, float(t))) for t in fine]
    best_t2, best_f2 = max(vals2, key=lambda x: x[1])
    log(f"Validation best macro F0.5={best_f2:.6f} at threshold={best_t2:.4f}")
    return best_t2, best_f2


def write_candidates(con: sqlite3.Connection, out_path: Path, chunk_size: int = 100_000) -> None:
    log(f"Writing {out_path} ...")
    cur = con.execute("SELECT s1_id,match_id FROM candidates ORDER BY s1_id,match_id")
    s1_cur = con.execute("SELECT entity_id FROM records WHERE source='S1' ORDER BY entity_id")
    next_c = cur.fetchone()
    next_s1 = s1_cur.fetchone()
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        while next_s1:
            s1 = next_s1[0]
            ids: List[str] = []
            while next_c and next_c[0] < s1:
                next_c = cur.fetchone()
            while next_c and next_c[0] == s1:
                ids.append(next_c[1])
                next_c = cur.fetchone()
            f.write(s1 + "\t" + ",".join(ids) + "\n")
            next_s1 = s1_cur.fetchone()


def write_predictions(con: sqlite3.Connection, pred_path: Path, threshold: float, model: lgb.LGBMClassifier, chunk_size: int = 150_000) -> None:
    con.execute("DROP TABLE IF EXISTS predictions")
    con.execute("CREATE TABLE predictions(s1_id TEXT NOT NULL, match_id TEXT NOT NULL, score REAL NOT NULL, PRIMARY KEY(s1_id,match_id))")
    con.execute("CREATE INDEX idx_predictions_s1 ON predictions(s1_id)")
    con.commit()

    last = 0
    total = 0
    accepted = 0
    while True:
        q = f"""
        SELECT c.rowid AS c_rowid, c.s1_id, c.match_id, c.block_mask,
               a.name_norm AS s1_name, b.name_norm AS m_name,
               a.name_core AS s1_name_core, b.name_core AS m_name_core,
               a.name_sorted AS s1_name_sorted, b.name_sorted AS m_name_sorted,
               a.name_first AS s1_name_first, b.name_first AS m_name_first,
               a.address_norm AS s1_address, b.address_norm AS m_address,
               a.address_sorted AS s1_address_sorted, b.address_sorted AS m_address_sorted,
               a.postal AS s1_postal, b.postal AS m_postal,
               a.house AS s1_house, b.house AS m_house,
               a.address_mid AS s1_mid, b.address_mid AS m_mid,
               b.source AS m_source
        FROM candidates c
        JOIN records a ON a.entity_id=c.s1_id AND a.source='S1'
        JOIN records b ON b.entity_id=c.match_id
        WHERE c.rowid>{last}
        ORDER BY c.rowid
        LIMIT {chunk_size}
        """
        df = pd.read_sql_query(q, con)
        if df.empty:
            break
        X = feature_frame(df)
        scores = model.predict_proba(X)[:,1]
        take = scores >= threshold
        if take.any():
            rows = [(str(s1), str(mid), float(sc)) for s1, mid, sc in zip(df.loc[take,"s1_id"], df.loc[take,"match_id"], scores[take])]
            con.executemany("INSERT OR REPLACE INTO predictions(s1_id,match_id,score) VALUES (?,?,?)", rows)
            con.commit()
            accepted += len(rows)
        total += len(df)
        last = int(df["c_rowid"].iloc[-1])
        if total % 500_000 < chunk_size:
            log(f"Inference: scored={total:,}, accepted={accepted:,}")

    log(f"Inference complete: scored={total:,}, accepted={accepted:,}")
    log(f"Writing {pred_path} ...")
    cur = con.execute("SELECT s1_id,match_id FROM predictions ORDER BY s1_id,match_id")
    s1_cur = con.execute("SELECT entity_id FROM records WHERE source='S1' ORDER BY entity_id")
    next_p = cur.fetchone()
    next_s1 = s1_cur.fetchone()
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        while next_s1:
            s1 = next_s1[0]
            ids: List[str] = []
            while next_p and next_p[0] < s1:
                next_p = cur.fetchone()
            while next_p and next_p[0] == s1:
                ids.append(next_p[1])
                next_p = cur.fetchone()
            f.write(s1 + "\t" + ",".join(ids) + "\n")
            next_s1 = s1_cur.fetchone()


def invoke_validator(validator: Path, output_dir: Path, test_dir: Path) -> int:
    cmd = [
        sys.executable, str(validator),
        "--matching", str(output_dir / "matching_results.tsv"),
        "--candidate", str(output_dir / "candidate_pairs.tsv"),
        "--test-dir", str(test_dir),
    ]
    log("Running submission validator...")
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd)


def build_train_db(
    train_dir: Path, work_dir: Path,
    chunk_size: int = 250_000,
    max_pos: int = 7_500_000,
    max_neg: int = 8_000_000,
    singleton_neg: int = 2_000_000,
    cache_gb: int = 12,
    mmap_gb: int = 32,
) -> Path:
    db_path = work_dir / "train_er.sqlite"
    con = open_db(db_path, cache_gb=cache_gb, mmap_gb=mmap_gb)
    init_record_db(con)
    log(f"CPU workers={CPU_WORKERS}; SQLite cache={cache_gb}GB; mmap={mmap_gb}GB")
    s1_ids_df = load_sources_into_db(con, train_dir, "train", chunk_size=chunk_size)
    load_ground_truth(con, train_dir / "train_ground_truth.tsv", s1_ids_df["entity_id"].tolist())
    generate_candidates(con)
    materialize_train_pairs(con, max_pos=max_pos, max_neg=max_neg, singleton_neg=singleton_neg)
    con.close()
    return db_path


def run_train(train_db: Path, model_path: Path, max_train_rows: int = 14_000_000, n_estimators: int = 1200) -> Tuple[lgb.LGBMClassifier, float, float]:
    con = sqlite3.connect(str(train_db))
    X, y = collect_training_arrays(con, max_rows=max_train_rows)
    model = train_model(X, y, n_estimators=n_estimators)
    true_counts, s1_idx, scores, labels = validation_arrays(con, model)
    threshold, val_f = tune_threshold(true_counts, s1_idx, scores, labels)
    # Save model and threshold using LightGBM's native text format.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    Path(str(model_path) + ".threshold").write_text(f"{threshold:.10f}\n{val_f:.10f}\n", encoding="utf-8")
    con.close()
    return model, threshold, val_f


def load_saved_model(model_path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(model_path))


def predict_with_booster(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(X[FEATURES])


def test_db(test_dir: Path, work_dir: Path, chunk_size: int = 250_000, cache_gb: int = 12, mmap_gb: int = 32) -> Path:
    db_path = work_dir / "test_er.sqlite"
    con = open_db(db_path, cache_gb=cache_gb, mmap_gb=mmap_gb)
    init_record_db(con)
    load_sources_into_db(con, test_dir, "test", chunk_size=chunk_size)
    # No ground truth exists for test, so candidates are generated only.
    generate_candidates_test(con)
    con.close()
    return db_path


def generate_candidates_test(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM candidates")
    con.commit()
    blocks = [
        ("key_name", "name_exact"),
        ("key_name_core", "name_core"),
        ("key_name_sorted", "name_sorted"),
        ("key_name_prefix", "name_prefix"),
        ("key_address", "address_exact"),
        ("key_address_sorted", "address_sorted"),
        ("key_postal_name", "postal_name"),
        ("key_house_name", "house_name"),
        ("key_house_mid", "house_mid"),
        ("key_postal_mid", "postal_mid"),
        ("key_house_postal", "house_postal"),
    ]
    for key, block_name in blocks:
        bit = BLOCK_BITS[block_name]
        where = ""
        if key == "key_name_prefix":
            where = "WHERE length(a.name_prefix)>=8 AND length(b.name_prefix)>=8"
        log(f"Test blocking on {block_name} ...")
        for source in ("S2", "S3"):
            add_block(con, source, key, bit, where)
        n = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        log(f"  test candidates so far: {n:,}")
    stats = con.execute("SELECT AVG(cnt),MAX(cnt),COUNT(*) FROM (SELECT s1_id,COUNT(*) cnt FROM candidates GROUP BY s1_id)").fetchone()
    total_s1 = con.execute("SELECT COUNT(*) FROM records WHERE source='S1'").fetchone()[0]
    log(f"Test candidate pairs={con.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]:,}; avg over candidate-bearing S1={stats[0] or 0:.2f}; max/S1={stats[1] or 0:,}; S1 total={total_s1:,}")


def run_test(test_db: Path, model_path: Path, threshold: float, output_dir: Path) -> None:
    con = sqlite3.connect(str(test_db))
    booster = load_saved_model(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Need a wrapper matching predict_proba-style output.
    class BoosterWrapper:
        def predict_proba(self, X):
            p = booster.predict(X[FEATURES])
            return np.column_stack([1-p,p])
    wrapper = BoosterWrapper()
    write_predictions(con, output_dir / "matching_results.tsv", threshold, wrapper)
    write_candidates(con, output_dir / "candidate_pairs.tsv")
    con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    ap.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--work-dir", type=Path, default=Path("work"))
    ap.add_argument("--model", type=Path, default=Path("work/entity_matcher.txt"))
    ap.add_argument("--validator", type=Path, default=None)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=250_000)
    ap.add_argument("--cache-gb", type=int, default=12)
    ap.add_argument("--mmap-gb", type=int, default=32)
    ap.add_argument("--max-pos", type=int, default=7_500_000)
    ap.add_argument("--max-neg", type=int, default=8_000_000)
    ap.add_argument("--singleton-neg", type=int, default=2_000_000)
    ap.add_argument("--max-train-rows", type=int, default=14_000_000)
    ap.add_argument("--n-estimators", type=int, default=1200)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required_train = [
        args.train_dir / "train_source1.tsv", args.train_dir / "train_source2.tsv",
        args.train_dir / "train_source3.tsv", args.train_dir / "train_ground_truth.tsv",
    ]
    required_test = [
        args.test_dir / "test_source1.tsv", args.test_dir / "test_source2.tsv", args.test_dir / "test_source3.tsv",
    ]
    missing = [str(p) for p in required_train + required_test if not p.exists()]
    if missing:
        log("Missing required dataset files:")
        for p in missing:
            log("  " + p)
        return 2

    log(f"Detected CPU workers: {CPU_WORKERS}")
    log(f"RAM target: 64GB-class machine; SQLite cache={args.cache_gb}GB, mmap={args.mmap_gb}GB")
    t0 = time.time()
    train_db = build_train_db(
        args.train_dir, args.work_dir,
        chunk_size=args.chunk_size,
        max_pos=args.max_pos,
        max_neg=args.max_neg,
        singleton_neg=args.singleton_neg,
        cache_gb=args.cache_gb,
        mmap_gb=args.mmap_gb,
    )
    model, threshold, val_f = run_train(
        train_db, args.model,
        max_train_rows=args.max_train_rows,
        n_estimators=args.n_estimators,
    )
    log(f"Training/validation completed in {(time.time()-t0)/60:.1f} min; threshold={threshold:.4f}, validation F0.5={val_f:.6f}")

    test_db_path = test_db(
        args.test_dir, args.work_dir,
        chunk_size=args.chunk_size,
        cache_gb=args.cache_gb,
        mmap_gb=args.mmap_gb,
    )
    run_test(test_db_path, args.model, threshold, args.output_dir)

    if args.validator:
        rc = invoke_validator(args.validator, args.output_dir, args.test_dir)
        if rc != 0:
            log(f"Validator returned exit code {rc}")
            return rc
        log("Validator PASS")

    if not args.keep_work:
        for p in [train_db, test_db_path, Path(str(train_db)+"-wal"), Path(str(train_db)+"-shm"), Path(str(test_db_path)+"-wal"), Path(str(test_db_path)+"-shm")]:
            try:
                if p.exists(): p.unlink()
            except OSError:
                pass

    log("DONE")
    log(f"matching_results.tsv: {args.output_dir / 'matching_results.tsv'}")
    log(f"candidate_pairs.tsv:  {args.output_dir / 'candidate_pairs.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
