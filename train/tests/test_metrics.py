"""Tests for train.metrics — the numbers that accuracy was hiding.

The motivating case has its own test (`test_all_majority_predictor_*`): on this
project's real class balance, answering "Not Out" every time scores 86%
accuracy. If macro-F1 does not collapse there, the metric is worthless and the
whole change is theatre.
"""

import torch

from train.metrics import (
    compute_metrics,
    confusion_matrix,
    format_confusion,
    metrics_from_confusion,
    summary_line,
)

# The real label balance, measured over 2010 R2 labels on 2026-07-30, scaled to
# a 100-frame val set: 86 Not Out / 6 Full / 8 Partial.
REAL_BALANCE = [0] * 86 + [1] * 6 + [2] * 8


class TestConfusionMatrix:
    def test_rows_are_truth_columns_are_predictions(self):
        # One true-Full frame predicted Not Out => row 1, column 0.
        matrix = confusion_matrix([1], [0])
        assert matrix[1][0] == 1
        assert sum(sum(row) for row in matrix) == 1

    def test_accepts_torch_tensors(self):
        targets = torch.tensor([0, 1, 2, 1])
        preds = torch.tensor([0, 1, 2, 0])
        assert confusion_matrix(targets, preds) == [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]

    def test_out_of_range_indices_are_dropped_not_fatal(self):
        # A telemetry probe must never be what kills an hour-long run.
        assert sum(sum(r) for r in confusion_matrix([0, 7], [0, 0])) == 1

    def test_empty_input_is_all_zeros(self):
        assert confusion_matrix([], []) == [[0] * 3 for _ in range(3)]


class TestPerfectPredictor:
    def test_everything_is_one(self):
        metrics = compute_metrics(REAL_BALANCE, REAL_BALANCE)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        for stats in metrics["per_class"].values():
            assert stats["precision"] == 1.0
            assert stats["recall"] == 1.0
            assert stats["f1"] == 1.0
        assert metrics["visible"]["precision"] == 1.0
        assert metrics["visible"]["recall"] == 1.0
        # Support adds up to the frames that went in.
        assert metrics["n"] == len(REAL_BALANCE)
        assert metrics["visible"]["support"] == 14


class TestAllMajorityPredictor:
    """THE case this whole change exists for."""

    METRICS = compute_metrics(REAL_BALANCE, [0] * len(REAL_BALANCE))

    def test_accuracy_looks_great_and_means_nothing(self):
        assert self.METRICS["accuracy"] == 0.86

    def test_macro_f1_is_far_below_accuracy(self):
        # 0.86 accuracy, 0.31 macro-F1: the gap IS the finding.
        assert self.METRICS["macro_f1"] < 0.35
        assert self.METRICS["accuracy"] - self.METRICS["macro_f1"] > 0.5

    def test_balanced_accuracy_is_chance(self):
        # One class right, two at zero recall => 1/3.
        assert abs(self.METRICS["balanced_accuracy"] - 1 / 3) < 0.001

    def test_the_mountain_is_never_detected(self):
        for name in ("full", "partial"):
            assert self.METRICS["per_class"][name]["recall"] == 0.0
            assert self.METRICS["per_class"][name]["f1"] == 0.0
        visible = self.METRICS["visible"]
        assert visible["recall"] == 0.0
        assert visible["predicted"] == 0
        # Zero predicted positives must not raise; precision is 0.0, not NaN.
        assert visible["precision"] == 0.0

    def test_not_out_recall_is_perfect_which_is_the_trap(self):
        assert self.METRICS["per_class"]["not_out"]["recall"] == 1.0
        assert self.METRICS["per_class"]["not_out"]["precision"] == 0.86


class TestPrecisionRecallMath:
    def test_matches_hand_computed_values(self):
        # true:  0 0 0 1 1 2
        # pred:  0 0 1 1 2 2
        targets = [0, 0, 0, 1, 1, 2]
        preds = [0, 0, 1, 1, 2, 2]
        metrics = compute_metrics(targets, preds)
        not_out = metrics["per_class"]["not_out"]
        assert not_out["precision"] == 1.0  # 2 predicted, 2 right
        assert round(not_out["recall"], 4) == round(2 / 3, 4)
        full = metrics["per_class"]["full"]
        assert full["precision"] == 0.5  # 2 predicted (one was a true Not Out)
        assert full["recall"] == 0.5
        partial = metrics["per_class"]["partial"]
        assert partial["precision"] == 0.5
        assert partial["recall"] == 1.0
        assert metrics["accuracy"] == round(4 / 6, 4)

    def test_f1_is_the_harmonic_mean(self):
        metrics = compute_metrics([1, 1, 0, 0], [1, 0, 0, 0])
        full = metrics["per_class"]["full"]
        expected = (
            2
            * full["precision"]
            * full["recall"]
            / (full["precision"] + full["recall"])
        )
        assert abs(full["f1"] - expected) < 1e-4


class TestVisibleBinaryView:
    def test_full_and_partial_confusion_is_not_a_visible_error(self):
        # Calling a Full frame Partial still means "the mountain is out", which
        # is the question the Worker's alert answers. Per-class F1 punishes it;
        # the binary view must not.
        metrics = compute_metrics([1, 2], [2, 1])
        assert metrics["accuracy"] == 0.0
        assert metrics["visible"]["precision"] == 1.0
        assert metrics["visible"]["recall"] == 1.0

    def test_false_positive_is_the_failure_the_project_cares_about(self):
        # Two Not Out frames announced as visible: 2 false positives.
        metrics = compute_metrics([0, 0, 0, 1], [1, 2, 0, 1])
        visible = metrics["visible"]
        assert visible["predicted"] == 3
        assert visible["support"] == 1
        assert round(visible["precision"], 4) == round(1 / 3, 4)
        assert visible["recall"] == 1.0


class TestDegenerateInputs:
    def test_class_absent_from_val_is_excluded_from_the_macro_average(self):
        # No Partial frames at all. Averaging a 0.0 for a class that was never
        # in the val set would report a model failure that did not happen.
        metrics = compute_metrics([0, 0, 1, 1], [0, 0, 1, 1])
        assert metrics["per_class"]["partial"]["support"] == 0
        assert metrics["macro_classes"] == ["not_out", "full"]
        assert metrics["macro_f1"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0

    def test_absent_class_that_gets_predicted_still_costs_precision(self):
        # Partial is not in truth but the model emits it: it must not vanish
        # silently — the frame it stole shows up as lost recall elsewhere.
        metrics = compute_metrics([0, 0, 1, 1], [0, 2, 1, 1])
        assert metrics["per_class"]["partial"]["support"] == 0
        assert metrics["per_class"]["partial"]["predicted"] == 1
        assert metrics["per_class"]["not_out"]["recall"] == 0.5
        assert metrics["macro_f1"] < 1.0

    def test_zero_predicted_positives_does_not_divide_by_zero(self):
        metrics = compute_metrics([1, 1], [0, 0])
        assert metrics["per_class"]["full"]["precision"] == 0.0
        assert metrics["per_class"]["not_out"]["precision"] == 0.0
        assert metrics["per_class"]["not_out"]["support"] == 0

    def test_empty_val_set_yields_zeros_not_nan(self):
        metrics = compute_metrics([], [])
        assert metrics["n"] == 0
        assert metrics["accuracy"] == 0.0
        assert metrics["macro_f1"] == 0.0
        assert metrics["balanced_accuracy"] == 0.0
        assert metrics["macro_classes"] == []

    def test_metrics_are_json_serialisable(self):
        # They travel through --json-summary and --progress-jsonl.
        import json

        assert json.loads(json.dumps(compute_metrics(REAL_BALANCE, REAL_BALANCE)))


class TestRendering:
    def test_confusion_is_a_readable_grid(self):
        text = format_confusion(
            metrics_from_confusion([[3, 1, 0], [0, 2, 1], [1, 0, 4]])
        )
        lines = text.splitlines()
        assert len(lines) == 4  # header + 3 truth rows
        assert "Not Out" in lines[1] and "3" in lines[1]
        assert lines[0].strip().startswith("true\\pred")

    def test_missing_metrics_degrade_not_crash(self):
        assert format_confusion(None) == "n/a"
        assert format_confusion({}) == "n/a"
        assert summary_line(None) == "no metrics"

    def test_summary_line_leads_with_the_ungameable_numbers(self):
        line = summary_line(compute_metrics(REAL_BALANCE, [0] * len(REAL_BALANCE)))
        assert line.startswith("macro_f1=0.308")
        assert "Full: P0.00/R0.00(n=6)" in line
