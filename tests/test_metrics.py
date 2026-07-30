"""Tests for the segmentation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (SegmentationEvaluator, average_precision, box_iou,
                         dice_coefficient, iou_matrix, mask_iou, match_instances)


def square_mask(size=10, x1=0, y1=0, x2=5, y2=5):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def test_mask_iou_identical_masks():
    mask = square_mask()
    assert mask_iou(mask, mask) == pytest.approx(1.0)


def test_mask_iou_disjoint_masks():
    assert mask_iou(square_mask(), square_mask(x1=6, y1=6, x2=10, y2=10)) == 0.0


def test_mask_iou_half_overlap():
    a = square_mask(10, 0, 0, 4, 4)     # 16 px
    b = square_mask(10, 2, 0, 6, 4)     # 16 px, 8 px shared
    assert mask_iou(a, b) == pytest.approx(8 / 24)


def test_mask_iou_of_two_empty_masks_is_zero():
    empty = np.zeros((5, 5), dtype=np.uint8)
    assert mask_iou(empty, empty) == 0.0


def test_mask_iou_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mask_iou(np.zeros((4, 4)), np.zeros((5, 5)))


def test_dice_is_higher_than_iou_for_partial_overlap():
    a = square_mask(10, 0, 0, 4, 4)
    b = square_mask(10, 2, 0, 6, 4)
    assert dice_coefficient(a, b) > mask_iou(a, b)
    assert dice_coefficient(a, b) == pytest.approx(2 * 8 / 32)


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_iou_matrix_shape_and_values():
    preds = np.stack([square_mask(10, 0, 0, 4, 4), square_mask(10, 6, 6, 10, 10)])
    gts = np.stack([square_mask(10, 0, 0, 4, 4)])

    matrix = iou_matrix(preds, gts)

    assert matrix.shape == (2, 1)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 0] == 0.0


def test_iou_matrix_with_no_predictions():
    assert iou_matrix(np.zeros((0, 5, 5)), np.ones((2, 5, 5))).shape == (0, 2)


def test_each_ground_truth_is_matched_at_most_once():
    """Two overlapping predictions on one target: one TP, one FP."""
    gt = np.stack([square_mask(10, 0, 0, 6, 6)])
    preds = np.stack([square_mask(10, 0, 0, 6, 6), square_mask(10, 0, 0, 5, 5)])

    matches, unmatched_pred, unmatched_gt = match_instances(
        preds, gt, scores=[0.9, 0.8], iou_threshold=0.5)

    assert len(matches) == 1
    assert matches[0][0] == 0        # highest scoring prediction wins
    assert unmatched_pred == [1]
    assert unmatched_gt == []


def test_low_iou_predictions_do_not_match():
    gt = np.stack([square_mask(10, 0, 0, 4, 4)])
    preds = np.stack([square_mask(10, 7, 7, 10, 10)])

    matches, unmatched_pred, unmatched_gt = match_instances(preds, gt, [0.99], 0.5)

    assert matches == []
    assert unmatched_pred == [0]
    assert unmatched_gt == [0]


def test_match_instances_rejects_mismatched_scores():
    with pytest.raises(ValueError):
        match_instances(np.zeros((2, 4, 4)), np.zeros((1, 4, 4)), scores=[0.5])


def test_average_precision_perfect_ranking():
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)


def test_average_precision_penalises_false_positive_ahead_of_hit():
    perfect = average_precision([0.9, 0.8], [True, True], 2)
    noisy = average_precision([0.9, 0.8], [False, True], 2)
    assert noisy < perfect


def test_average_precision_without_ground_truth_is_zero():
    assert average_precision([0.9], [False], 0) == 0.0


def test_evaluator_perfect_predictions():
    gt = np.stack([square_mask(10, 0, 0, 5, 5)])
    evaluator = SegmentationEvaluator(iou_threshold=0.5)
    evaluator.add(gt.copy(), [0.95], gt)

    result = evaluator.compute()

    assert result['precision'] == pytest.approx(1.0)
    assert result['recall'] == pytest.approx(1.0)
    assert result['f1'] == pytest.approx(1.0)
    assert result['mean_iou'] == pytest.approx(1.0)
    assert result['ap@0.5'] == pytest.approx(1.0)


def test_evaluator_counts_misses_and_false_alarms():
    evaluator = SegmentationEvaluator(iou_threshold=0.5)
    gt = np.stack([square_mask(10, 0, 0, 5, 5)])

    evaluator.add(gt.copy(), [0.9], gt)                       # hit
    evaluator.add(np.zeros((0, 10, 10)), [], gt)              # missed detection
    evaluator.add(np.stack([square_mask(10, 6, 6, 10, 10)]), [0.7],
                  np.zeros((0, 10, 10)))                      # false alarm

    result = evaluator.compute()

    assert result['true_positives'] == 1
    assert result['false_negatives'] == 1
    assert result['false_positives'] == 1
    assert result['precision'] == pytest.approx(0.5)
    assert result['recall'] == pytest.approx(0.5)
    assert result['images'] == 3


def test_evaluator_with_no_data_is_all_zeros():
    result = SegmentationEvaluator().compute()
    assert result['precision'] == 0.0
    assert result['recall'] == 0.0
    assert result['f1'] == 0.0
