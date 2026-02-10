"""
Comprehensive Examples: Object Detection Architectures

This file demonstrates all object detection models with practical examples.

Author: Ali Mehdi
Date: January 31, 2026
"""

import numpy as np
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

print("=" * 80)
print("OBJECT DETECTION ARCHITECTURES - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: YOLO - Real-Time Detection
# ============================================================================
print("EXAMPLE 1: YOLO - Real-Time Object Detection")
print("-" * 80)

yolo = YOLO(num_classes=80, input_size=416, grid_size=13)

print("YOLO configuration:")
print(f"Classes: {yolo.num_classes} (COCO dataset)")
print(f"Input size: {yolo.input_size}×{yolo.input_size}")
print(f"Grid size: {yolo.grid_size}×{yolo.grid_size}")
print(f"Anchors per cell: {yolo.num_anchors}")
print()

# Simulate image
image = np.random.randn(1, 3, 416, 416)

print(f"Input image: {image.shape}")

# Detect objects
boxes, scores, classes = yolo.detect(image, conf_threshold=0.5, nms_threshold=0.5)

print(f"Detected objects: {len(boxes)}")
if len(boxes) > 0:
    print(f"Bounding boxes: {boxes.shape}")
    print(f"Confidence scores: {scores.shape}")
    print(f"Class IDs: {classes.shape}")
print()

print("YOLO advantages:")
print("✓ Real-time speed (30-60 FPS)")
print("✓ End-to-end training")
print("✓ Good for video processing")
print("✓ Single forward pass")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Faster R-CNN - High Accuracy Detection
# ============================================================================
print("EXAMPLE 2: Faster R-CNN - High Accuracy Detection")
print("-" * 80)

rcnn = FasterRCNN(num_classes=20, backbone='resnet50', rpn_nms_threshold=0.7)

print("Faster R-CNN configuration:")
print(f"Classes: {rcnn.num_classes} (VOC dataset)")
print(f"Backbone: {rcnn.backbone}")
print(f"RPN NMS threshold: {rcnn.rpn_nms_threshold}")
print()

# Simulate image
image = np.random.randn(1, 3, 800, 800)

print(f"Input image: {image.shape}")

# Generate proposals
proposals = rcnn.generate_proposals(
    feature_map_size=(25, 25),
    image_size=(800, 800),
    num_proposals=300
)

print(f"Region proposals: {len(proposals)}")
print()

# Detect objects
boxes, scores, classes = rcnn.forward(image, conf_threshold=0.5)

print(f"Detected objects: {len(boxes)}")
print()

print("Faster R-CNN advantages:")
print("✓ High accuracy")
print("✓ Precise localization")
print("✓ Good for dense scenes")
print("✓ Two-stage refinement")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: SSD - Fast Multi-Scale Detection
# ============================================================================
print("EXAMPLE 3: SSD - Fast Multi-Scale Detection")
print("-" * 80)

ssd = SSD(num_classes=21, input_size=300, feature_scales=[38, 19, 10, 5, 3, 1])

print("SSD configuration:")
print(f"Classes: {ssd.num_classes}")
print(f"Input size: {ssd.input_size}×{ssd.input_size}")
print(f"Feature scales: {ssd.feature_scales}")
print()

# Simulate image
image = np.random.randn(1, 3, 300, 300)

print(f"Input image: {image.shape}")

# Detect objects
boxes, scores, classes = ssd.detect(image, conf_threshold=0.5, nms_threshold=0.45)

print(f"Detected objects: {len(boxes)}")
print()

print("SSD advantages:")
print("✓ Very fast (50+ FPS)")
print("✓ Multi-scale detection")
print("✓ Good for edge devices")
print("✓ Lightweight")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: RetinaNet - Balanced Detection with Focal Loss
# ============================================================================
print("EXAMPLE 4: RetinaNet - Balanced Detection with Focal Loss")
print("-" * 80)

retinanet = RetinaNet(
    num_classes=80,
    backbone='resnet50',
    focal_alpha=0.25,
    focal_gamma=2.0
)

print("RetinaNet configuration:")
print(f"Classes: {retinanet.num_classes}")
print(f"Backbone: {retinanet.backbone}")
print(f"Focal alpha: {retinanet.focal_alpha}")
print(f"Focal gamma: {retinanet.focal_gamma}")
print(f"FPN levels: {retinanet.fpn_levels}")
print()

# Simulate image
image = np.random.randn(1, 3, 800, 800)

# Detect objects
boxes, scores, classes = retinanet.detect(image, conf_threshold=0.5)

print(f"Detected objects: {len(boxes)}")
print()

# Focal loss example
predictions = np.random.randn(100)
targets = np.random.randint(0, 2, 100)
focal_loss = retinanet.focal_loss(predictions, targets)

print(f"Focal loss: {focal_loss:.4f}")
print()

print("RetinaNet advantages:")
print("✓ Balanced speed/accuracy")
print("✓ Handles class imbalance")
print("✓ Good for small objects")
print("✓ Feature Pyramid Network")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Anchor Generation
# ============================================================================
print("EXAMPLE 5: Anchor Box Generation")
print("-" * 80)

generator = AnchorGenerator(
    scales=[32, 64, 128, 256, 512],
    aspect_ratios=[0.5, 1.0, 2.0]
)

print("Anchor generator configuration:")
print(f"Scales: {generator.scales}")
print(f"Aspect ratios: {generator.aspect_ratios}")
print()

# Generate anchors
anchors = generator.generate(
    feature_map_size=(13, 13),
    image_size=(416, 416)
)

print(f"Generated anchors: {len(anchors)}")
print(f"Anchor shape: {anchors.shape}")
print(f"Anchors per location: {len(generator.scales) * len(generator.aspect_ratios)}")
print()

print("First 3 anchors:")
for i in range(min(3, len(anchors))):
    print(f"  Anchor {i}: {anchors[i]}")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: IoU Computation
# ============================================================================
print("EXAMPLE 6: Intersection over Union (IoU)")
print("-" * 80)

box1 = np.array([10, 10, 50, 50])
box2 = np.array([30, 30, 70, 70])

print("Box 1: [x1=10, y1=10, x2=50, y2=50]")
print("Box 2: [x1=30, y1=30, x2=70, y2=70]")
print()

iou = compute_iou(box1, box2)

print(f"IoU: {iou:.4f}")
print()

print("IoU interpretation:")
print("  0.0 - 0.3: Poor overlap")
print("  0.3 - 0.5: Moderate overlap")
print("  0.5 - 0.7: Good overlap")
print("  0.7 - 1.0: Excellent overlap")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Non-Maximum Suppression (NMS)
# ============================================================================
print("EXAMPLE 7: Non-Maximum Suppression (NMS)")
print("-" * 80)

# Overlapping boxes
boxes = np.array([
    [10, 10, 50, 50],
    [15, 15, 55, 55],
    [12, 12, 52, 52],
    [100, 100, 150, 150],
    [105, 105, 155, 155]
])

scores = np.array([0.9, 0.85, 0.8, 0.95, 0.88])

print(f"Input boxes: {len(boxes)}")
print(f"Scores: {scores}")
print()

# Apply NMS
keep = non_maximum_suppression(boxes, scores, iou_threshold=0.5)

print(f"After NMS: {len(keep)} boxes")
print(f"Kept indices: {keep}")
print()

print("NMS removes:")
print("✓ Duplicate detections")
print("✓ Overlapping boxes (IoU > threshold)")
print("✓ Keeps highest confidence boxes")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: mAP Evaluation
# ============================================================================
print("EXAMPLE 8: Mean Average Precision (mAP)")
print("-" * 80)

# Predictions
pred_boxes = [
    np.array([[10, 10, 50, 50], [100, 100, 150, 150]]),
    np.array([[20, 20, 60, 60]])
]

pred_scores = [
    np.array([0.9, 0.85]),
    np.array([0.8])
]

# Ground truth
gt_boxes = [
    np.array([[12, 12, 52, 52], [102, 102, 152, 152]]),
    np.array([[22, 22, 62, 62]])
]

print(f"Images: {len(pred_boxes)}")
print(f"Total predictions: {sum(len(p) for p in pred_boxes)}")
print(f"Total ground truth: {sum(len(g) for g in gt_boxes)}")
print()

# Compute mAP
map_score = compute_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)

print(f"mAP@0.5: {map_score:.4f}")
print()

print("mAP interpretation:")
print("  0.0 - 0.3: Poor detector")
print("  0.3 - 0.5: Moderate detector")
print("  0.5 - 0.7: Good detector")
print("  0.7 - 1.0: Excellent detector")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Comparing Detectors
# ============================================================================
print("EXAMPLE 9: Comparing Object Detectors")
print("-" * 80)

print("Performance comparison:")
print()

print("YOLO:")
print("  Speed: ★★★★★ (30-60 FPS)")
print("  Accuracy: ★★★☆☆")
print("  Use case: Real-time video, autonomous driving")
print()

print("Faster R-CNN:")
print("  Speed: ★★☆☆☆ (5-10 FPS)")
print("  Accuracy: ★★★★★")
print("  Use case: High-accuracy tasks, medical imaging")
print()

print("SSD:")
print("  Speed: ★★★★★ (50+ FPS)")
print("  Accuracy: ★★★☆☆")
print("  Use case: Edge devices, mobile, embedded")
print()

print("RetinaNet:")
print("  Speed: ★★★☆☆ (10-15 FPS)")
print("  Accuracy: ★★★★☆")
print("  Use case: Balanced tasks, small objects")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Real-World Application - Autonomous Driving
# ============================================================================
print("EXAMPLE 10: Real-World Application - Autonomous Driving")
print("-" * 80)

print("Autonomous driving detection pipeline:")
print()

# Use YOLO for real-time detection
yolo = YOLO(num_classes=80, input_size=416)

print("Step 1: Capture frame from camera")
frame = np.random.randn(1, 3, 416, 416)
print(f"  Frame: {frame.shape}")
print()

print("Step 2: Detect objects (YOLO)")
boxes, scores, classes = yolo.detect(frame, conf_threshold=0.5)
print(f"  Detected: {len(boxes)} objects")
print()

print("Step 3: Filter relevant classes")
# Class IDs: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
relevant_classes = [0, 2, 3, 5, 7]
mask = np.isin(classes, relevant_classes)
filtered_boxes = boxes[mask] if len(boxes) > 0 else boxes
print(f"  Relevant objects: {len(filtered_boxes)}")
print()

print("Step 4: Track objects across frames")
print("  (Use Kalman filter or SORT algorithm)")
print()

print("Step 5: Make driving decisions")
print("  ✓ Brake if pedestrian detected")
print("  ✓ Change lane if obstacle ahead")
print("  ✓ Adjust speed based on traffic")

print("\n✓ Example 10 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ YOLO - Real-Time Detection")
print("2. ✓ Faster R-CNN - High Accuracy")
print("3. ✓ SSD - Fast Multi-Scale")
print("4. ✓ RetinaNet - Balanced with Focal Loss")
print("5. ✓ Anchor Generation")
print("6. ✓ IoU Computation")
print("7. ✓ Non-Maximum Suppression")
print("8. ✓ mAP Evaluation")
print("9. ✓ Comparing Detectors")
print("10. ✓ Autonomous Driving Application")
print()
print("You now have a complete understanding of object detection!")
print()
print("Next steps:")
print("- Use YOLO for real-time applications")
print("- Use Faster R-CNN for high accuracy")
print("- Use SSD for edge devices")
print("- Use RetinaNet for balanced performance")
print("- Apply to autonomous driving, surveillance, retail")
