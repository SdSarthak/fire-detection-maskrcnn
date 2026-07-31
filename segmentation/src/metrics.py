"""
Instance-segmentation metrics for the fire detection model.

Pure NumPy so the evaluation logic can be unit tested without a GPU, a
trained checkpoint, or a deep learning framework in the loop.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary masks of identical shape."""
    a, b = _as_bool(a), _as_bool(b)
    if a.shape != b.shape:
        raise ValueError(f'Mask shape mismatch: {a.shape} vs {b.shape}')
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Dice / F1 overlap of two binary masks."""
    a, b = _as_bool(a), _as_bool(b)
    if a.shape != b.shape:
        raise ValueError(f'Mask shape mismatch: {a.shape} vs {b.shape}')
    total = a.sum() + b.sum()
    if total == 0:
        return 0.0
    return float(2.0 * np.logical_and(a, b).sum() / total)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU of two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(P, H, W)`` and ``(G, H, W)`` mask stacks."""
    pred_masks = np.asarray(pred_masks)
    gt_masks = np.asarray(gt_masks)
    if pred_masks.size == 0 or gt_masks.size == 0:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)

    if pred_masks.shape[1:] != gt_masks.shape[1:]:
        raise ValueError(
            'Predicted and ground-truth masks must share a spatial shape; got '
            f'{pred_masks.shape[1:]} vs {gt_masks.shape[1:]}. Predictions are '
            'reported at the original image size, so the targets must be too.')

    preds = _as_bool(pred_masks).reshape(len(pred_masks), -1)
    gts = _as_bool(gt_masks).reshape(len(gt_masks), -1)

    intersection = preds.astype(np.float64) @ gts.astype(np.float64).T
    pred_area = preds.sum(axis=1, keepdims=True).astype(np.float64)
    gt_area = gts.sum(axis=1, keepdims=True).astype(np.float64).T
    union = pred_area + gt_area - intersection
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(union > 0, intersection / union, 0.0)
    return result


def match_instances(pred_masks: np.ndarray,
                    gt_masks: np.ndarray,
                    scores: Sequence[float] = None,
                    iou_threshold: float = 0.5
                    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedily match predictions to ground truth by descending score.

    Returns ``(matches, unmatched_pred_indices, unmatched_gt_indices)`` where
    each match is ``(pred_index, gt_index, iou)``. Every ground-truth instance
    is claimed at most once, which is what makes the TP/FP/FN counts honest.
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if scores is None:
        scores = np.ones(n_pred, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != n_pred:
        raise ValueError('scores must have one entry per predicted mask')

    ious = iou_matrix(pred_masks, gt_masks)
    order = np.argsort(-scores) if n_pred else np.empty(0, dtype=int)

    matches: List[Tuple[int, int, float]] = []
    unmatched_pred: List[int] = []
    claimed = set()

    for pred_index in order:
        best_gt, best_iou = -1, 0.0
        for gt_index in range(n_gt):
            if gt_index in claimed:
                continue
            iou = float(ious[pred_index, gt_index])
            if iou > best_iou:
                best_gt, best_iou = gt_index, iou
        if best_gt >= 0 and best_iou >= iou_threshold:
            claimed.add(best_gt)
            matches.append((int(pred_index), int(best_gt), best_iou))
        else:
            unmatched_pred.append(int(pred_index))

    unmatched_gt = [i for i in range(n_gt) if i not in claimed]
    return matches, sorted(unmatched_pred), unmatched_gt


def average_precision(scores: Sequence[float], true_positives: Sequence[bool],
                      num_ground_truth: int) -> float:
    """All-point interpolated average precision over a score-ranked list."""
    if num_ground_truth <= 0:
        return 0.0
    if len(scores) == 0:
        return 0.0

    scores = np.asarray(scores, dtype=np.float64)
    flags = np.asarray(true_positives, dtype=bool)
    order = np.argsort(-scores)
    flags = flags[order]

    tp_cumulative = np.cumsum(flags)
    fp_cumulative = np.cumsum(~flags)

    recall = tp_cumulative / float(num_ground_truth)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)

    # Make precision monotonically decreasing, then integrate over recall.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0]], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


class SegmentationEvaluator:
    """Accumulate per-image detections and report dataset-level metrics."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = float(iou_threshold)
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError(
                f'iou_threshold must be in (0, 1]; got {self.iou_threshold}')
        self.reset()

    def reset(self) -> None:
        self._scores: List[float] = []
        self._true_positives: List[bool] = []
        self._matched_ious: List[float] = []
        self._matched_dice: List[float] = []
        self.num_ground_truth = 0
        self.num_predictions = 0
        self.num_images = 0

    def add(self, pred_masks: np.ndarray, scores: Sequence[float],
            gt_masks: np.ndarray) -> None:
        """Add one image worth of predictions and ground truth."""
        pred_masks = np.asarray(pred_masks)
        gt_masks = np.asarray(gt_masks)
        scores = [float(s) for s in scores]

        if len(scores) != len(pred_masks):
            raise ValueError(
                f'Got {len(scores)} score(s) for {len(pred_masks)} predicted '
                'mask(s); they must line up one-to-one.')
        if any(not np.isfinite(s) for s in scores):
            raise ValueError('Detection scores must be finite; got NaN or inf.')

        matches, _, _ = match_instances(
            pred_masks, gt_masks, scores, self.iou_threshold)

        matched_preds = {pred_index for pred_index, _, _ in matches}
        for pred_index in range(len(pred_masks)):
            self._scores.append(float(scores[pred_index]))
            self._true_positives.append(pred_index in matched_preds)

        for pred_index, gt_index, iou in matches:
            self._matched_ious.append(iou)
            self._matched_dice.append(
                dice_coefficient(pred_masks[pred_index], gt_masks[gt_index]))

        self.num_ground_truth += len(gt_masks)
        self.num_predictions += len(pred_masks)
        self.num_images += 1

    def compute(self) -> Dict[str, float]:
        """Precision / recall / F1 / mean IoU / mean Dice / AP for the dataset."""
        true_positives = int(np.sum(self._true_positives))
        false_positives = self.num_predictions - true_positives
        false_negatives = self.num_ground_truth - true_positives

        precision = (true_positives / self.num_predictions
                     if self.num_predictions else 0.0)
        recall = (true_positives / self.num_ground_truth
                  if self.num_ground_truth else 0.0)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            'images': float(self.num_images),
            'true_positives': float(true_positives),
            'false_positives': float(false_positives),
            'false_negatives': float(false_negatives),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'mean_iou': float(np.mean(self._matched_ious)) if self._matched_ious else 0.0,
            'mean_dice': float(np.mean(self._matched_dice)) if self._matched_dice else 0.0,
            f'ap@{self.iou_threshold:g}': average_precision(
                self._scores, self._true_positives, self.num_ground_truth),
        }
