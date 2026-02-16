"""
Object Detection: Recommended Defaults & Model Comparison

This example demonstrates how to use recommended defaults for each model
and compare models fairly with consistent settings.

Author: Ali Mehdi
Date: February 16, 2026
"""

import numpy as np
from ilovetools.ml.detection import (
    YOLO,
    FasterRCNN,
    SSD,
    RetinaNet,
    get_recommended_defaults,
    print_all_defaults,
    compare_models,
)

print("=" * 80)
print("OBJECT DETECTION: RECOMMENDED DEFAULTS & MODEL COMPARISON")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: View All Recommended Defaults
# ============================================================================
print("EXAMPLE 1: View All Recommended Defaults")
print("-" * 80)

print_all_defaults()

print("✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Using Recommended Defaults with YOLO
# ============================================================================
print("EXAMPLE 2: Using Recommended Defaults with YOLO")
print("-" * 80)

# Get recommended defaults for YOLO
yolo_defaults = get_recommended_defaults('YOLO')

print("YOLO Recommended Settings:")
print(f"  Confidence Threshold: {yolo_defaults['conf_threshold']}")
print(f"  NMS Threshold: {yolo_defaults['nms_threshold']}")
print(f"  Use Case: {yolo_defaults['use_case']}")
print(f"  Speed: {yolo_defaults['speed']}")
print(f"  Accuracy: {yolo_defaults['accuracy']}")
print()

# Create YOLO detector
yolo = YOLO(num_classes=80, input_size=416)
image = np.random.randn(1, 3, 416, 416)

# Option 1: Use recommended defaults automatically (None = use defaults)
print("Option 1: Automatic defaults (pass None)")
boxes, scores, classes = yolo.detect(image, conf_threshold=None, nms_threshold=None)
print(f"  Detected: {len(boxes)} objects")
print()

# Option 2: Explicitly use recommended defaults
print("Option 2: Explicit defaults")
boxes, scores, classes = yolo.detect(
    image,
    conf_threshold=yolo_defaults['conf_threshold'],
    nms_threshold=yolo_defaults['nms_threshold']
)
print(f"  Detected: {len(boxes)} objects")
print()

# Option 3: Custom thresholds (override defaults)
print("Option 3: Custom thresholds")
boxes, scores, classes = yolo.detect(
    image,
    conf_threshold=0.5,  # Higher than default 0.25
    nms_threshold=0.3    # Lower than default 0.45
)
print(f"  Detected: {len(boxes)} objects (stricter filtering)")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Comparing All Models with Recommended Defaults
# ============================================================================
print("EXAMPLE 3: Comparing All Models with Recommended Defaults")
print("-" * 80)

# Create all detectors
yolo = YOLO(num_classes=80, input_size=416)
rcnn = FasterRCNN(num_classes=20, backbone='resnet50')
ssd = SSD(num_classes=21, input_size=300)
retinanet = RetinaNet(num_classes=80)

# Test images
yolo_image = np.random.randn(1, 3, 416, 416)
rcnn_image = np.random.randn(1, 3, 800, 800)
ssd_image = np.random.randn(1, 3, 300, 300)
retinanet_image = np.random.randn(1, 3, 800, 800)

print("Running all models with recommended defaults:")
print()

# YOLO
boxes, scores, classes = yolo.detect(yolo_image)  # Uses defaults automatically
print(f"YOLO: {len(boxes)} detections")
print(f"  Settings: conf={yolo.recommended['conf_threshold']}, nms={yolo.recommended['nms_threshold']}")
print()

# Faster R-CNN
boxes, scores, classes = rcnn.forward(rcnn_image)  # Uses defaults automatically
print(f"Faster R-CNN: {len(boxes)} detections")
print(f"  Settings: conf={rcnn.recommended['conf_threshold']}, nms={rcnn.recommended['nms_threshold']}")
print()

# SSD
boxes, scores, classes = ssd.detect(ssd_image)  # Uses defaults automatically
print(f"SSD: {len(boxes)} detections")
print(f"  Settings: conf={ssd.recommended['conf_threshold']}, nms={ssd.recommended['nms_threshold']}")
print()

# RetinaNet
boxes, scores, classes = retinanet.detect(retinanet_image)  # Uses defaults automatically
print(f"RetinaNet: {len(boxes)} detections")
print(f"  Settings: conf={retinanet.recommended['conf_threshold']}, nms={retinanet.recommended['nms_threshold']}")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Model Comparison by Metric
# ============================================================================
print("EXAMPLE 4: Model Comparison by Metric")
print("-" * 80)

# Compare speed
print("Speed Comparison:")
speed_comparison = compare_models('speed')
for model, speed in speed_comparison.items():
    print(f"  {model}: {speed}")
print()

# Compare accuracy
print("Accuracy Comparison:")
accuracy_comparison = compare_models('accuracy')
for model, accuracy in accuracy_comparison.items():
    print(f"  {model}: {accuracy}")
print()

# Compare use cases
print("Use Case Comparison:")
use_case_comparison = compare_models('use_case')
for model, use_case in use_case_comparison.items():
    print(f"  {model}: {use_case}")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Impact of Threshold Changes
# ============================================================================
print("EXAMPLE 5: Impact of Threshold Changes")
print("-" * 80)

yolo = YOLO(num_classes=80, input_size=416)
image = np.random.randn(1, 3, 416, 416)

print("Testing different confidence thresholds (NMS=0.45):")
for conf in [0.1, 0.25, 0.5, 0.7, 0.9]:
    boxes, scores, classes = yolo.detect(image, conf_threshold=conf, nms_threshold=0.45)
    print(f"  conf={conf}: {len(boxes)} detections")
print()

print("Testing different NMS thresholds (conf=0.25):")
for nms in [0.1, 0.3, 0.45, 0.6, 0.8]:
    boxes, scores, classes = yolo.detect(image, conf_threshold=0.25, nms_threshold=nms)
    print(f"  nms={nms}: {len(boxes)} detections")
print()

print("Key Insights:")
print("✓ Higher conf_threshold → Fewer detections (more precision, less recall)")
print("✓ Lower conf_threshold → More detections (less precision, more recall)")
print("✓ Higher nms_threshold → More overlapping boxes kept")
print("✓ Lower nms_threshold → Stricter duplicate removal")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Choosing the Right Model
# ============================================================================
print("EXAMPLE 6: Choosing the Right Model for Your Use Case")
print("-" * 80)

print("Decision Guide:")
print()

print("1. Real-time video processing (30+ FPS needed):")
print("   → Use YOLO or SSD")
print("   → YOLO defaults: conf=0.25, nms=0.45")
print("   → SSD defaults: conf=0.5, nms=0.45")
print()

print("2. High accuracy required (medical imaging, critical tasks):")
print("   → Use Faster R-CNN")
print("   → Defaults: conf=0.7, nms=0.3 (high precision)")
print()

print("3. Edge device deployment (mobile, embedded):")
print("   → Use SSD")
print("   → Defaults: conf=0.5, nms=0.45 (balanced)")
print()

print("4. Small object detection:")
print("   → Use RetinaNet")
print("   → Defaults: conf=0.5, nms=0.5 (focal loss helps)")
print()

print("5. Balanced speed/accuracy:")
print("   → Use RetinaNet or YOLO")
print("   → RetinaNet: Better accuracy, slower")
print("   → YOLO: Faster, good accuracy")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Fair Model Comparison
# ============================================================================
print("EXAMPLE 7: Fair Model Comparison (Same Test Conditions)")
print("-" * 80)

print("To compare models fairly:")
print()

print("1. Use recommended defaults for each model")
print("   ✓ Each model has optimized settings")
print("   ✓ Reflects real-world performance")
print()

print("2. Test on same dataset")
print("   ✓ Same images, same ground truth")
print("   ✓ Compute mAP for each model")
print()

print("3. Measure both speed and accuracy")
print("   ✓ FPS (frames per second)")
print("   ✓ mAP (mean Average Precision)")
print()

print("4. Consider your constraints")
print("   ✓ Real-time requirement → YOLO/SSD")
print("   ✓ Accuracy requirement → Faster R-CNN")
print("   ✓ Resource constraint → SSD")
print()

print("Example comparison table:")
print()
print("Model         | FPS    | mAP   | Recommended For")
print("-" * 60)
print("YOLO          | 30-60  | 0.65  | Real-time video")
print("Faster R-CNN  | 5-10   | 0.85  | High accuracy")
print("SSD           | 50+    | 0.60  | Edge devices")
print("RetinaNet     | 10-15  | 0.75  | Balanced tasks")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Accessing Model Defaults Programmatically
# ============================================================================
print("EXAMPLE 8: Accessing Model Defaults Programmatically")
print("-" * 80)

# Get all defaults
from ilovetools.ml.detection import RECOMMENDED_DEFAULTS

print("All recommended defaults:")
for model_name, defaults in RECOMMENDED_DEFAULTS.items():
    print(f"\n{model_name}:")
    for key, value in defaults.items():
        print(f"  {key}: {value}")

print("\n✓ Example 8 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Key Takeaways:")
print()
print("1. ✓ Each model has research-based recommended defaults")
print("2. ✓ Use None to automatically apply recommended defaults")
print("3. ✓ Override defaults when you have specific requirements")
print("4. ✓ Compare models fairly using their recommended settings")
print("5. ✓ Confidence threshold affects precision/recall tradeoff")
print("6. ✓ NMS threshold controls duplicate removal strictness")
print("7. ✓ Choose model based on speed/accuracy requirements")
print()
print("Quick Reference:")
print("  YOLO: conf=0.25, nms=0.45 (real-time)")
print("  Faster R-CNN: conf=0.7, nms=0.3 (high accuracy)")
print("  SSD: conf=0.5, nms=0.45 (edge devices)")
print("  RetinaNet: conf=0.5, nms=0.5 (balanced)")
