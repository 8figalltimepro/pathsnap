"""Evaluation metrics for candidate classification and route reconstruction."""
from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from .candidates import CandidateSet


def candidate_probability_metrics(cand: CandidateSet, probs: np.ndarray,
                                  true_arc: np.ndarray, valid: np.ndarray) -> dict:
    """Candidate-level binary classification quality (labels: is this the true arc?)."""
    if len(cand) == 0 or valid.sum() == 0:
        return {"macro_f1": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan"),
                "log_loss": float("nan"), "n_candidate_rows": 0}
    mask = valid[cand.obs_idx]
    y = (cand.arc == true_arc[cand.obs_idx])[mask].astype(np.int64)
    p = np.clip(probs[mask].astype(np.float64), 1e-7, 1 - 1e-7)
    yhat = (p >= 0.5).astype(np.int64)
    return {
        "macro_f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "f1": float(f1_score(y, yhat, average="binary", zero_division=0)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "n_candidate_rows": int(mask.sum()),
    }


def true_candidate_log_loss(cand: CandidateSet, probs: np.ndarray,
                            true_arc: np.ndarray, valid: np.ndarray) -> float:
    """Mean negative log probability assigned to the reference arc (top-1 ranking quality)."""
    if len(cand) == 0 or valid.sum() == 0:
        return float("nan")
    sub = np.isin(cand.obs_idx, np.where(valid)[0])
    idx = np.where(sub & (cand.arc == true_arc[cand.obs_idx]))[0]
    if len(idx) == 0:
        return float("nan")
    p = np.clip(probs[idx], 1e-7, 1.0)
    return float(-np.log(p).mean())


def candidate_miss_rate(cand: CandidateSet, true_arc: np.ndarray,
                        valid: np.ndarray) -> float:
    """Fraction of scored fixes whose reference arc is missing from the candidates."""
    if valid.sum() == 0 or len(cand) == 0:
        return float("nan")
    hit = np.zeros(len(true_arc), dtype=bool)
    sub = np.isin(cand.obs_idx, np.where(valid)[0])
    is_true = cand.arc[sub] == true_arc[cand.obs_idx[sub]]
    hit[cand.obs_idx[sub][is_true]] = True
    return float(1.0 - hit[valid].mean())


def top1_accuracy(pred_arc: np.ndarray, true_arc: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() == 0:
        return float("nan")
    return float((pred_arc[valid] == true_arc[valid]).mean())


def route_continuity(net, pred_arc: np.ndarray) -> float:
    """Fraction of consecutive predictions that stay on connected arcs."""
    if len(pred_arc) < 2:
        return float("nan")
    a = pred_arc[:-1]
    b = pred_arc[1:]
    arcs = net.trip.arcs
    conn = (a == b) | (arcs[b, 0] == arcs[a, 1])
    return float(conn.mean())


def compress_arcs(arc: np.ndarray) -> list[int]:
    out = []
    prev = None
    for a in arc:
        a = int(a)
        if a != prev:
            out.append(a)
            prev = a
    return out


def sequence_f1(pred_arc: np.ndarray, ref_route) -> dict:
    """Sequence-level precision/recall/F1 between predicted and reference arc paths."""
    if ref_route is None or len(ref_route) == 0:
        return {"seq_precision": float("nan"), "seq_recall": float("nan"),
                "seq_f1": float("nan")}
    p = compress_arcs(pred_arc)
    r = [int(x) for x in ref_route]
    if not p or not r:
        return {"seq_precision": float("nan"), "seq_recall": float("nan"),
                "seq_f1": float("nan")}
    sm = SequenceMatcher(a=p, b=r, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    prec = matched / len(p)
    rec = matched / len(r)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"seq_precision": float(prec), "seq_recall": float(rec), "seq_f1": float(f1)}


def compute_metrics(net, cand: CandidateSet, probs: np.ndarray,
                    pred_clf: np.ndarray, pred_hmm: np.ndarray,
                    true_arc: np.ndarray, labels_valid: np.ndarray,
                    ref_route) -> dict:
    """Full evaluation bundle for classifier-only and HMM route predictions."""
    out: dict = {
        "n_obs": int(len(true_arc)),
        "n_scored": int(labels_valid.sum()),
        "scored_fraction": float(labels_valid.mean()) if len(true_arc) else 0.0,
        "candidate_miss_rate": candidate_miss_rate(cand, true_arc, labels_valid),
        "top1_accuracy": top1_accuracy(pred_clf, true_arc, labels_valid),
        "top1_accuracy_hmm": top1_accuracy(pred_hmm, true_arc, labels_valid),
        "route_continuity": route_continuity(net, pred_clf),
        "route_continuity_hmm": route_continuity(net, pred_hmm),
        "true_candidate_log_loss": true_candidate_log_loss(cand, probs, true_arc, labels_valid),
    }
    out.update(candidate_probability_metrics(cand, probs, true_arc, labels_valid))
    out.update({f"{k}_clf": v for k, v in sequence_f1(pred_clf, ref_route).items()})
    out.update({f"{k}_hmm": v for k, v in sequence_f1(pred_hmm, ref_route).items()})
    return out
