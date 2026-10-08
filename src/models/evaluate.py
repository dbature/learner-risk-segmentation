"""Metrics and figures for the validation report.

Two families of metric, because the business question has two halves:

* threshold-free ranking quality: ROC AUC and average precision (the area
  under the precision-recall curve, the better guide when the classes are
  unbalanced, Saito and Rehmsmeier 2015)
* the operating point a coach actually uses: flag the riskiest 25% of a cohort
  (capacity constraint from Module 1), then report recall, precision, F1 and
  accuracy at that cut-off
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CAPACITY = 0.25
INK, NAVY, GREEN, RED, AMBER, GREY = "#1B2A3A", "#2E5C8A", "#1F7A5C", "#B3402F", "#C98A1B", "#6B7A8C"
PALETTE = [NAVY, GREEN, AMBER, RED, GREY]


def capacity_threshold(scores: np.ndarray, capacity: float = CAPACITY) -> float:
    """Score above which the top `capacity` share of learners is flagged."""
    return float(np.quantile(scores, 1 - capacity))


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "flag_rate": float(pred.mean()),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def _style(ax, title):
    ax.set_title(title, fontsize=11, color=INK, loc="left", fontweight="bold")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def plot_roc_pr(results: dict[str, tuple[np.ndarray, np.ndarray]], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for (name, (y, p)), c in zip(results.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=c, lw=2, label=f"{name}  AUC {roc_auc_score(y, p):.3f}")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, color=c, lw=2, label=f"{name}  AP {average_precision_score(y, p):.3f}")
    y0 = next(iter(results.values()))[0]
    axes[0].plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    axes[1].axhline(y0.mean(), ls="--", color=GREY, lw=1)
    _style(axes[0], "ROC curve, test set")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (recall)")
    _style(axes[1], "Precision-recall curve, test set")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False, loc="lower right" if ax is axes[0] else "upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_confusion(m: dict, title: str, out: Path) -> Path:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm, cmap="Blues")
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
    ax.set_yticks([0, 1], ["Completed", "Did not complete"])
    _style(ax, title)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_calibration(y: np.ndarray, p: np.ndarray, out: Path) -> Path:
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    ax.plot(mean, frac, marker="o", color=NAVY, lw=2)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed non-completion rate")
    _style(ax, "Calibration, test set (deciles)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_capacity(y: np.ndarray, p: np.ndarray, out: Path) -> Path:
    caps = np.linspace(0.05, 0.6, 23)
    rec = [recall_score(y, (p >= capacity_threshold(p, c)).astype(int)) for c in caps]
    prec = [precision_score(y, (p >= capacity_threshold(p, c)).astype(int)) for c in caps]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(caps * 100, np.array(rec) * 100, color=NAVY, lw=2, label="Recall (share of non-completers caught)")
    ax.plot(caps * 100, np.array(prec) * 100, color=GREEN, lw=2, label="Precision (share of flags that are right)")
    ax.plot(caps * 100, caps * 100, color=GREY, ls="--", lw=1, label="Recall if flags were random")
    ax.axvline(25, color=AMBER, lw=1.5)
    ax.text(25.6, 8, "coach capacity, 25%", color=AMBER, fontsize=9)
    ax.set_xlabel("Share of cohort flagged (%)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, "Capacity trade-off, test set")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
