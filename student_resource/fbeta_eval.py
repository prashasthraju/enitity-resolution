"""
Macro-averaged F0.5 scorer, matching the competition spec exactly.

Per source1 entity, with pred/truth as sets of matched_entity_ids:
    truth empty, pred empty      -> 1.0   (correct singleton)
    truth empty, pred non-empty  -> 0.0   (false merge on a singleton)
    truth non-empty, pred empty  -> 0.0   (recall = 0)
    overlap == 0 (both non-empty)-> 0.0
    else -> F0.5 = 1.25*P*R / (0.25*P + R)

The final score is the mean of per-entity F0.5 over every source1 entity
(macro average) -- NOT computed from pooled TP/FP/FN counts.
"""

from typing import Dict, Iterable


def _parse(cell) -> set:
    cell = (cell or "").strip()
    return set() if cell == "" else {x.strip() for x in cell.split(",")}


def per_entity_f05(pred: Iterable[str], truth: Iterable[str]) -> float:
    pred, truth = set(pred), set(truth)
    if not truth and not pred:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(truth)
    beta2 = 0.25  # beta = 0.5 -> beta^2 = 0.25
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def score_submission(pred_df, truth_df) -> Dict:
    """
    pred_df, truth_df: DataFrames with columns
        source1_entity_id, matched_entity_ids  (comma-separated string, "" if none)
    truth_df's entities must all be present in pred_df (as the real
    validator requires); raises if any are missing so you catch it before
    you burn a leaderboard submission on a formatting bug.
    """
    truth_map = dict(zip(truth_df["source1_entity_id"], truth_df["matched_entity_ids"]))
    pred_map = dict(zip(pred_df["source1_entity_id"], pred_df["matched_entity_ids"]))

    missing = set(truth_map) - set(pred_map)
    if missing:
        raise ValueError(f"{len(missing)} source1 entities missing from predictions, e.g. {list(missing)[:3]}")

    per_entity = {
        s1_id: per_entity_f05(_parse(pred_map.get(s1_id, "")), _parse(truth_cell))
        for s1_id, truth_cell in truth_map.items()
    }
    macro = sum(per_entity.values()) / len(per_entity) if per_entity else 0.0
    return {"per_entity": per_entity, "macro_f0_5": macro}


if __name__ == "__main__":
    # quick self-test against the worked example in the problem statement
    p = per_entity_f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    print(f"worked example: {p:.3f} (spec says 0.714)")
    assert abs(p - 0.714) < 0.001
