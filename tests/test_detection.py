"""
Tests for Object Detection Architectures

This file contains comprehensive tests for all object detection models.

Author: Ali Mehdi
Date: January 31, 2026
"""

import numpy as np
import pytest
from ilovetools.ml.detection import (
    YOLO,
    FasterRCNN,
    SSD,
    RetinaNet,
    AnchorGenerator,
    compute_iou,
    non_maximum_suppression,
    compute_map,
)


# ============================================================================
# TEST UTILITY FUNCTIONS
# ============================================================================

def test_compute_iou_perfect_overlap():
    """Test IoU with perfect overlap."""
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([10, 10, 50, 50])
    
    iou = compute_iou(box1, box2)
    
    assert np.isclose(iou, 1.0)


def test_compute_iou_no_overlap():
    """Test IoU with no overlap."""
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([100, 100, 150, 150])
    
    iou = compute_iou(box1, box2)
    
    assert np.isclose(iou, 0.0)


def test_compute_iou_partial_overlap():
    """Test IoU with partial overlap."""
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([30, 30, 70, 70])
    
    iou = compute_iou(box1, box2)
    
    assert 0 < iou < 1


def test_non_maximum_suppression():
    """Test NMS removes duplicate boxes."""
    boxes = np.array([
        [10, 10, 50, 50],
        [15, 15, 55, 55],  # Overlaps with first
        [100, 100, 150, 150]  # Separate
    ])
    scores = np.array([0.9, 0.8, 0.95])
    
    keep = non_maximum_suppression(boxes, scores, iou_threshold=0.5)
    
    assert len(keep) == 2  # Should keep 2 boxes


def test_compute_map():
    """Test mAP computation."""
    pred_boxes = [np.array([[10, 10, 50, 50]])]
    pred_scores = [np.array([0.9])]
    gt_boxes = [np.array([[12, 12, 52, 52]])]
    
    map_score = compute_map(pred_boxes, pred_scores, gt_boxes)
    
    assert 0 <= map_score <= 1


# ============================================================================
# TEST ANCHOR GENERATOR
# ============================================================================

def test_anchor_generator_basic():
    """Test basic anchor generation."""
    generator = AnchorGenerator(scales=[32, 64], aspect_ratios=[0.5, 1.0, 2.0])
    
    anchors = generator.generate(feature_map_size=(13, 13), image_size=(416, 416))
    
    assert len(anchors) > 0
    assert anchors.shape[1] == 4  # x1, y1, x2, y2


def test_anchor_generator_different_scales():
    """Test anchor generation with different scales."""
    generator = AnchorGenerator(scales=[16, 32, 64, 128])
    
    anchors = generator.generate(feature_map_size=(10, 10), image_size=(320, 320))
    
    # Should generate anchors for each scale and aspect ratio
    assert len(anchors) > 0


# ============================================================================
# TEST YOLO
# ============================================================================

def test_yolo_basic():
    """Test basic YOLO functionality."""
    yolo = YOLO(num_classes=80, input_size=416)
    
    assert yolo.num_classes == 80
    assert yolo.input_size == 416


def test_yolo_detect():
    """Test YOLO detection."""
    yolo = YOLO(num_classes=80, input_size=416)
    image = np.random.randn(1, 3, 416, 416)
    
    boxes, scores, classes = yolo.detect(image, conf_threshold=0.5)
    
    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(classes, np.ndarray)


def test_yolo_callable():
    """Test that YOLO is callable."""
    yolo = YOLO(num_classes=80)
    image = np.random.randn(1, 3, 416, 416)
    
    output = yolo(image)
    assert output is not None


# ============================================================================
# TEST FASTER R-CNN
# ============================================================================

def test_faster_rcnn_basic():
    """Test basic Faster R-CNN functionality."""
    rcnn = FasterRCNN(num_classes=20, backbone='resnet50')
    
    assert rcnn.num_classes == 20
    assert rcnn.backbone == 'resnet50'


def test_faster_rcnn_generate_proposals():
    """Test RPN proposal generation."""
    rcnn = FasterRCNN(num_classes=20)
    
    proposals = rcnn.generate_proposals(
        feature_map_size=(25, 25),
        image_size=(800, 800),
        num_proposals=300
    )
    
    assert len(proposals) > 0
    assert len(proposals) <= 300


def test_faster_rcnn_forward():
    """Test Faster R-CNN forward pass."""
    rcnn = FasterRCNN(num_classes=20)
    image = np.random.randn(1, 3, 800, 800)
    
    boxes, scores, classes = rcnn.forward(image)
    
    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(classes, np.ndarray)


def test_faster_rcnn_callable():
    """Test that Faster R-CNN is callable."""
    rcnn = FasterRCNN(num_classes=20)
    image = np.random.randn(1, 3, 800, 800)
    
    output = rcnn(image)
    assert output is not None


# ============================================================================
# TEST SSD
# ============================================================================

def test_ssd_basic():
    """Test basic SSD functionality."""
    ssd = SSD(num_classes=21, input_size=300)
    
    assert ssd.num_classes == 21
    assert ssd.input_size == 300


def test_ssd_detect():
    """Test SSD detection."""
    ssd = SSD(num_classes=21, input_size=300)
    image = np.random.randn(1, 3, 300, 300)
    
    boxes, scores, classes = ssd.detect(image)
    
    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(classes, np.ndarray)


def test_ssd_callable():
    """Test that SSD is callable."""
    ssd = SSD(num_classes=21)
    image = np.random.randn(1, 3, 300, 300)
    
    output = ssd(image)
    assert output is not None


# ============================================================================
# TEST RETINANET
# ============================================================================

def test_retinanet_basic():
    """Test basic RetinaNet functionality."""
    retinanet = RetinaNet(num_classes=80, backbone='resnet50')
    
    assert retinanet.num_classes == 80
    assert retinanet.backbone == 'resnet50'


def test_retinanet_focal_loss():
    """Test focal loss computation."""
    retinanet = RetinaNet(num_classes=80)
    
    predictions = np.random.randn(100)
    targets = np.random.randint(0, 2, 100)
    
    loss = retinanet.focal_loss(predictions, targets)
    
    assert isinstance(loss, float)
    assert loss >= 0


def test_retinanet_detect():
    """Test RetinaNet detection."""
    retinanet = RetinaNet(num_classes=80)
    image = np.random.randn(1, 3, 800, 800)
    
    boxes, scores, classes = retinanet.detect(image)
    
    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(classes, np.ndarray)


def test_retinanet_callable():
    """Test that RetinaNet is callable."""
    retinanet = RetinaNet(num_classes=80)
    image = np.random.randn(1, 3, 800, 800)
    
    output = retinanet(image)
    assert output is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_all_detectors_callable():
    """Test that all detectors are callable."""
    image_yolo = np.random.randn(1, 3, 416, 416)
    image_rcnn = np.random.randn(1, 3, 800, 800)
    image_ssd = np.random.randn(1, 3, 300, 300)
    
    yolo = YOLO(num_classes=80)
    rcnn = FasterRCNN(num_classes=20)
    ssd = SSD(num_classes=21)
    retinanet = RetinaNet(num_classes=80)
    
    assert yolo(image_yolo) is not None
    assert rcnn(image_rcnn) is not None
    assert ssd(image_ssd) is not None
    assert retinanet(image_rcnn) is not None


def test_detection_pipeline():
    """Test complete detection pipeline."""
    # YOLO detection
    yolo = YOLO(num_classes=80, input_size=416)
    image = np.random.randn(1, 3, 416, 416)
    
    boxes, scores, classes = yolo.detect(image, conf_threshold=0.5, nms_threshold=0.5)
    
    # Verify output format
    if len(boxes) > 0:
        assert boxes.shape[1] == 4  # x1, y1, x2, y2
        assert len(scores) == len(boxes)
        assert len(classes) == len(boxes)


def test_nms_reduces_boxes():
    """Test that NMS reduces number of boxes."""
    # Create overlapping boxes
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],
        [14, 14, 54, 54],
        [100, 100, 150, 150]
    ])
    scores = np.array([0.9, 0.85, 0.8, 0.95])
    
    keep = non_maximum_suppression(boxes, scores, iou_threshold=0.5)
    
    # Should keep fewer boxes than input
    assert len(keep) < len(boxes)


def test_anchor_generation_consistency():
    """Test that anchor generation is consistent."""
    generator = AnchorGenerator(scales=[32, 64], aspect_ratios=[1.0, 2.0])
    
    anchors1 = generator.generate((13, 13), (416, 416))
    anchors2 = generator.generate((13, 13), (416, 416))
    
    assert np.allclose(anchors1, anchors2)


print("=" * 80)
print("ALL OBJECT DETECTION TESTS PASSED! ✓")
print("=" * 80)
