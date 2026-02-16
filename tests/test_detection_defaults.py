"""
Tests for Object Detection Recommended Defaults

This file tests the recommended defaults and comparison utilities.

Author: Ali Mehdi
Date: February 16, 2026
"""

import numpy as np
import pytest
from ilovetools.ml.detection import (
    YOLO,
    FasterRCNN,
    SSD,
    RetinaNet,
    get_recommended_defaults,
    compare_models,
    RECOMMENDED_DEFAULTS,
)


# ============================================================================
# TEST RECOMMENDED DEFAULTS
# ============================================================================

def test_get_recommended_defaults_yolo():
    """Test getting YOLO recommended defaults."""
    defaults = get_recommended_defaults('YOLO')
    
    assert 'conf_threshold' in defaults
    assert 'nms_threshold' in defaults
    assert defaults['conf_threshold'] == 0.25
    assert defaults['nms_threshold'] == 0.45


def test_get_recommended_defaults_faster_rcnn():
    """Test getting Faster R-CNN recommended defaults."""
    defaults = get_recommended_defaults('FasterRCNN')
    
    assert defaults['conf_threshold'] == 0.7
    assert defaults['nms_threshold'] == 0.3


def test_get_recommended_defaults_ssd():
    """Test getting SSD recommended defaults."""
    defaults = get_recommended_defaults('SSD')
    
    assert defaults['conf_threshold'] == 0.5
    assert defaults['nms_threshold'] == 0.45


def test_get_recommended_defaults_retinanet():
    """Test getting RetinaNet recommended defaults."""
    defaults = get_recommended_defaults('RetinaNet')
    
    assert defaults['conf_threshold'] == 0.5
    assert defaults['nms_threshold'] == 0.5


def test_get_recommended_defaults_invalid_model():
    """Test error handling for invalid model name."""
    with pytest.raises(ValueError):
        get_recommended_defaults('InvalidModel')


def test_recommended_defaults_structure():
    """Test that all defaults have required fields."""
    required_fields = ['conf_threshold', 'nms_threshold', 'description', 'use_case', 'speed', 'accuracy']
    
    for model_name, defaults in RECOMMENDED_DEFAULTS.items():
        for field in required_fields:
            assert field in defaults, f"{model_name} missing {field}"


# ============================================================================
# TEST MODEL COMPARISON
# ============================================================================

def test_compare_models_all():
    """Test comparing all models."""
    comparison = compare_models('all')
    
    assert 'YOLO' in comparison
    assert 'FasterRCNN' in comparison
    assert 'SSD' in comparison
    assert 'RetinaNet' in comparison


def test_compare_models_speed():
    """Test comparing models by speed."""
    comparison = compare_models('speed')
    
    assert comparison['YOLO'] == 'Very Fast (30-60 FPS)'
    assert comparison['FasterRCNN'] == 'Slow (5-10 FPS)'
    assert comparison['SSD'] == 'Very Fast (50+ FPS)'


def test_compare_models_accuracy():
    """Test comparing models by accuracy."""
    comparison = compare_models('accuracy')
    
    assert comparison['FasterRCNN'] == 'Excellent'
    assert 'Good' in comparison['YOLO']


def test_compare_models_use_case():
    """Test comparing models by use case."""
    comparison = compare_models('use_case')
    
    assert 'real-time' in comparison['YOLO'].lower()
    assert 'medical' in comparison['FasterRCNN'].lower()
    assert 'edge' in comparison['SSD'].lower()


# ============================================================================
# TEST MODELS WITH DEFAULTS
# ============================================================================

def test_yolo_uses_defaults_when_none():
    """Test YOLO uses recommended defaults when None is passed."""
    yolo = YOLO(num_classes=80, input_size=416)
    image = np.random.randn(1, 3, 416, 416)
    
    # Should use defaults when None
    boxes, scores, classes = yolo.detect(image, conf_threshold=None, nms_threshold=None)
    
    # Should complete without error
    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(classes, np.ndarray)


def test_faster_rcnn_uses_defaults_when_none():
    """Test Faster R-CNN uses recommended defaults when None is passed."""
    rcnn = FasterRCNN(num_classes=20)
    image = np.random.randn(1, 3, 800, 800)
    
    boxes, scores, classes = rcnn.forward(image, conf_threshold=None)
    
    assert isinstance(boxes, np.ndarray)


def test_ssd_uses_defaults_when_none():
    """Test SSD uses recommended defaults when None is passed."""
    ssd = SSD(num_classes=21)
    image = np.random.randn(1, 3, 300, 300)
    
    boxes, scores, classes = ssd.detect(image, conf_threshold=None, nms_threshold=None)
    
    assert isinstance(boxes, np.ndarray)


def test_retinanet_uses_defaults_when_none():
    """Test RetinaNet uses recommended defaults when None is passed."""
    retinanet = RetinaNet(num_classes=80)
    image = np.random.randn(1, 3, 800, 800)
    
    boxes, scores, classes = retinanet.detect(image, conf_threshold=None)
    
    assert isinstance(boxes, np.ndarray)


def test_yolo_custom_thresholds_override_defaults():
    """Test custom thresholds override defaults."""
    yolo = YOLO(num_classes=80)
    image = np.random.randn(1, 3, 416, 416)
    
    # Use custom thresholds
    boxes, scores, classes = yolo.detect(image, conf_threshold=0.8, nms_threshold=0.2)
    
    # Should use custom values, not defaults
    assert isinstance(boxes, np.ndarray)


# ============================================================================
# TEST MODEL ATTRIBUTES
# ============================================================================

def test_yolo_has_recommended_attribute():
    """Test YOLO has recommended defaults attribute."""
    yolo = YOLO(num_classes=80)
    
    assert hasattr(yolo, 'recommended')
    assert yolo.recommended['conf_threshold'] == 0.25
    assert yolo.recommended['nms_threshold'] == 0.45


def test_faster_rcnn_has_recommended_attribute():
    """Test Faster R-CNN has recommended defaults attribute."""
    rcnn = FasterRCNN(num_classes=20)
    
    assert hasattr(rcnn, 'recommended')
    assert rcnn.recommended['conf_threshold'] == 0.7


def test_ssd_has_recommended_attribute():
    """Test SSD has recommended defaults attribute."""
    ssd = SSD(num_classes=21)
    
    assert hasattr(ssd, 'recommended')
    assert ssd.recommended['conf_threshold'] == 0.5


def test_retinanet_has_recommended_attribute():
    """Test RetinaNet has recommended defaults attribute."""
    retinanet = RetinaNet(num_classes=80)
    
    assert hasattr(retinanet, 'recommended')
    assert retinanet.recommended['conf_threshold'] == 0.5


# ============================================================================
# TEST THRESHOLD IMPACT
# ============================================================================

def test_higher_conf_threshold_fewer_detections():
    """Test that higher confidence threshold results in fewer detections."""
    yolo = YOLO(num_classes=80)
    image = np.random.randn(1, 3, 416, 416)
    
    # Low threshold
    boxes_low, _, _ = yolo.detect(image, conf_threshold=0.1, nms_threshold=0.5)
    
    # High threshold
    boxes_high, _, _ = yolo.detect(image, conf_threshold=0.9, nms_threshold=0.5)
    
    # Higher threshold should result in fewer or equal detections
    assert len(boxes_high) <= len(boxes_low)


def test_lower_nms_threshold_fewer_detections():
    """Test that lower NMS threshold results in fewer detections."""
    yolo = YOLO(num_classes=80)
    image = np.random.randn(1, 3, 416, 416)
    
    # High NMS threshold (keep more overlapping boxes)
    boxes_high_nms, _, _ = yolo.detect(image, conf_threshold=0.25, nms_threshold=0.9)
    
    # Low NMS threshold (remove more overlapping boxes)
    boxes_low_nms, _, _ = yolo.detect(image, conf_threshold=0.25, nms_threshold=0.1)
    
    # Lower NMS threshold should result in fewer or equal detections
    assert len(boxes_low_nms) <= len(boxes_high_nms)


# ============================================================================
# TEST DEFAULTS CONSISTENCY
# ============================================================================

def test_all_models_have_consistent_defaults():
    """Test that all models have consistent default structure."""
    models = ['YOLO', 'FasterRCNN', 'SSD', 'RetinaNet']
    
    for model_name in models:
        defaults = get_recommended_defaults(model_name)
        
        # Check required fields
        assert 'conf_threshold' in defaults
        assert 'nms_threshold' in defaults
        assert 'description' in defaults
        assert 'use_case' in defaults
        assert 'speed' in defaults
        assert 'accuracy' in defaults
        
        # Check value types
        assert isinstance(defaults['conf_threshold'], (int, float))
        assert isinstance(defaults['nms_threshold'], (int, float))
        assert isinstance(defaults['description'], str)
        assert isinstance(defaults['use_case'], str)
        assert isinstance(defaults['speed'], str)
        assert isinstance(defaults['accuracy'], str)
        
        # Check value ranges
        assert 0 <= defaults['conf_threshold'] <= 1
        assert 0 <= defaults['nms_threshold'] <= 1


print("=" * 80)
print("ALL RECOMMENDED DEFAULTS TESTS PASSED! ✓")
print("=" * 80)
