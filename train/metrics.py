"""Per-class evaluation metrics — the numbers accuracy hides.

The label set is severely imbalanced (measured 2026-07-30 over 2010 R2 labels:
1735 Not Out / 164 Partial / 111 Full). A model that answers "Not Out" to every
frame scores **86.3% accuracy**, so a headline accuracy near 99% says almost
nothing about the only question this project actually asks — *is the mountain
out?* Worse, the repo's stated constraint is "precision over recall: minimize
false positives (announcing the mountain is out when it isn't)", and raw
accuracy cannot express that at all.

So this module computes, from a confusion matrix:

* per-class precision / recall / F1 / support,
* **macro-F1** and **balanced accuracy** — averages over classes, which the
  all-majority predictor cannot game (it scores 0.31 / 0.33 where accuracy
  scores 0.86),
* a derived **"visible" binary view** (Full+Partial vs Not Out), because that
  collapse is the product question and exactly what the Worker's alerts key on.

Implemented with plain torch/numpy-free arithmetic on purpose: scikit-learn is a
`dev`-group dependency only, and the trainer must not grow a main dependency for
four divisions.

Conventions: class indices follow `bot/labeler.py` — 0=Not Out, 1=Full,
2=Partial. Confusion matrix rows are TRUE classes, columns are PREDICTED.
Every rate is guarded against a zero denominator and reports 0.0 rather than
NaN, so a class the model never predicts renders as "0% precision", not a crash.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Mirrors bot/labeler.py CLASS_NAMES / CLASS_LABELS — kept as a local constant so
# train/ never has to import the bot (discord.py is a separate dependency group).
CLASS_NAMES = ["not_out", "full", "partial"]
CLASS_LABELS = ["Not Out", "Full", "Partial"]
NUM_CLASSES = 3

# The classes that mean "you can see the mountain". The Worker alerts on this
# collapse, so it gets first-class precision/recall of its own.
VISIBLE_CLASSES = (1, 2)


def _to_int_list(values: Any) -> list[int]:
    """Accept a torch tensor, numpy array, or any sequence of ints."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, int):
        return [values]
    return [int(v) for v in values]


def confusion_matrix(
    targets: Any, preds: Any, num_classes: int = NUM_CLASSES
) -> list[list[int]]:
    """Rows = true class, columns = predicted class.

    Out-of-range indices are dropped rather than raising: a metrics probe must
    never be the thing that kills an hour-long training run.
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]
    target_list = _to_int_list(targets)
    pred_list = _to_int_list(preds)
    for true_c, pred_c in zip(target_list, pred_list, strict=True):
        if 0 <= true_c < num_classes and 0 <= pred_c < num_classes:
            matrix[true_c][pred_c] += 1
    return matrix


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def metrics_from_confusion(matrix: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Every headline number this project should be judged on, from one matrix.

    Macro-F1 and balanced accuracy average over the classes **present in the
    val set** (support > 0). Averaging over an absent class would drag both
    numbers toward zero for a reason that has nothing to do with the model, and
    a val split that lost a class entirely is a split bug to fix, not a score to
    report. `macro_classes` names what was averaged, so the number is never
    silently computed over a different denominator than the reader assumes.
    """
    num_classes = len(matrix)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[c][c] for c in range(num_classes))

    per_class: dict[str, dict[str, float]] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    macro_classes: list[str] = []

    for c in range(num_classes):
        tp = matrix[c][c]
        fn = sum(matrix[c]) - tp
        fp = sum(matrix[r][c] for r in range(num_classes)) - tp
        support = tp + fn
        stats = _prf(tp, fp, fn)
        stats["support"] = support
        stats["predicted"] = tp + fp
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        per_class[name] = stats
        if support > 0:
            recalls.append(stats["recall"])
            f1s.append(stats["f1"])
            macro_classes.append(name)

    # Binary collapse: positive = "visible" (Full or Partial).
    vis_tp = sum(
        matrix[t][p]
        for t in VISIBLE_CLASSES
        for p in VISIBLE_CLASSES
        if t < num_classes
    )
    vis_fn = sum(sum(matrix[t]) for t in VISIBLE_CLASSES if t < num_classes) - vis_tp
    vis_fp = sum(
        matrix[t][p]
        for t in range(num_classes)
        for p in VISIBLE_CLASSES
        if t not in VISIBLE_CLASSES
    )
    visible = _prf(vis_tp, vis_fp, vis_fn)
    visible["support"] = vis_tp + vis_fn
    visible["predicted"] = vis_tp + vis_fp

    return {
        "accuracy": round(_safe_div(correct, total), 4),
        "macro_f1": round(_safe_div(sum(f1s), len(f1s)), 4),
        "balanced_accuracy": round(_safe_div(sum(recalls), len(recalls)), 4),
        "macro_classes": macro_classes,
        "per_class": per_class,
        "visible": visible,
        "confusion": [list(row) for row in matrix],
        "n": total,
    }


def compute_metrics(
    targets: Any, preds: Any, num_classes: int = NUM_CLASSES
) -> dict[str, Any]:
    """Convenience wrapper: predictions + truth in, the full metric block out."""
    return metrics_from_confusion(confusion_matrix(targets, preds, num_classes))


def format_confusion(metrics: dict[str, Any] | None) -> str:
    """A 3x3 matrix as fixed-width text.

    Nine Discord embed fields would blow the embed budget and read worse than
    the table does; callers wrap this in a code block.
    """
    matrix = (metrics or {}).get("confusion")
    if not matrix:
        return "n/a"
    header = "true\\pred  " + "".join(f"{label:>9}" for label in CLASS_LABELS)
    lines = [header]
    for c, row in enumerate(matrix):
        label = CLASS_LABELS[c] if c < len(CLASS_LABELS) else f"class_{c}"
        lines.append(f"{label:<10}" + "".join(f"{value:>9}" for value in row))
    return "\n".join(lines)


def summary_line(metrics: dict[str, Any] | None) -> str:
    """One-line console rendering for the epoch log."""
    if not metrics:
        return "no metrics"
    per_class = metrics.get("per_class", {})
    parts = [
        f"macro_f1={metrics.get('macro_f1', 0.0):.3f}",
        f"bal_acc={metrics.get('balanced_accuracy', 0.0):.1%}",
    ]
    for name, label in zip(CLASS_NAMES, CLASS_LABELS, strict=True):
        stats = per_class.get(name)
        if not stats:
            continue
        parts.append(
            f"{label}: P{stats['precision']:.2f}/R{stats['recall']:.2f}"
            f"(n={stats['support']})"
        )
    visible = metrics.get("visible")
    if visible:
        parts.append(f"visible: P{visible['precision']:.2f}/R{visible['recall']:.2f}")
    return "  ".join(parts)
