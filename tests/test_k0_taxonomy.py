"""Unit tests for Iteration K.0 Failure Taxonomy Harness."""

import numpy as np
import pytest
from tools.benchmark_k0_detector import FailureTaxonomy, compute_iou_xyxy


def test_iou_computation():
    boxA = np.array([0, 0, 10, 10], dtype=np.float32)
    boxB = np.array([0, 0, 10, 10], dtype=np.float32)
    assert abs(compute_iou_xyxy(boxA, boxB) - 1.0) < 1e-5

    boxC = np.array([5, 0, 15, 10], dtype=np.float32)
    # intersection: [5, 0, 10, 10] -> area 50, union: 100 + 100 - 50 = 150 -> 1/3
    assert abs(compute_iou_xyxy(boxA, boxC) - (1.0 / 3.0)) < 1e-5

    boxD = np.array([20, 20, 30, 30], dtype=np.float32)
    assert compute_iou_xyxy(boxA, boxD) == 0.0


def test_taxonomy_enum_definitions():
    assert FailureTaxonomy.CAT_1_NO_DETECTION.value == "CAT_1_NO_DETECTION"
    assert FailureTaxonomy.CAT_2_SUB_THRESHOLD.value == "CAT_2_SUB_THRESHOLD"
    assert FailureTaxonomy.CAT_3_WRONG_CLASS.value == "CAT_3_WRONG_CLASS"
    assert FailureTaxonomy.CAT_4_DEFORMED_BBOX.value == "CAT_4_DEFORMED_BBOX"
    assert FailureTaxonomy.CAT_5_INTERMITTENT.value == "CAT_5_INTERMITTENT"
    assert FailureTaxonomy.SUCCESS.value == "SUCCESS"
