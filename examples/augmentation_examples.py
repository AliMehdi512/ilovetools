"""
Comprehensive Examples: Data Augmentation

This file demonstrates all augmentation techniques with practical examples and use cases.

Author: Ali Mehdi
Date: January 15, 2026
"""

import numpy as np
from ilovetools.ml.augmentation import (
    ImageAugmenter,
    TextAugmenter,
    AugmentationPipeline
)

print("=" * 80)
print("DATA AUGMENTATION - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Basic Image Augmentation
# ============================================================================
print("EXAMPLE 1: Basic Image Augmentation")
print("-" * 80)

aug = ImageAugmenter(seed=42)

# Create sample image
image = np.random.rand(224, 224, 3)
print(f"Original image shape: {image.shape}")

# Horizontal flip
flipped = aug.horizontal_flip(image)
print(f"Flipped image shape: {flipped.shape}")

# Rotation
rotated = aug.random_rotation(image, max_angle=30)
print(f"Rotated image shape: {rotated.shape}")

# Random crop
cropped = aug.random_crop(image, crop_size=(128, 128))
print(f"Cropped image shape: {cropped.shape}")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Color Augmentation
# ============================================================================
print("EXAMPLE 2: Color Augmentation")
print("-" * 80)

aug = ImageAugmenter(seed=42)
image = np.random.rand(100, 100, 3)

# Color jitter
jittered = aug.color_jitter(image, brightness=0.3, contrast=0.3, saturation=0.3)
print(f"Color jittered image range: [{jittered.min():.3f}, {jittered.max():.3f}]")

# Gaussian noise
noisy = aug.gaussian_noise(image, std=0.05)
print(f"Noisy image range: [{noisy.min():.3f}, {noisy.max():.3f}]")

# Gaussian blur
blurred = aug.gaussian_blur(image, sigma=2.0)
print(f"Blurred image shape: {blurred.shape}")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Advanced Augmentation - Mixup
# ============================================================================
print("EXAMPLE 3: Advanced Augmentation - Mixup")
print("-" * 80)

aug = ImageAugmenter(seed=42)

# Two different images
image1 = np.random.rand(100, 100, 3)
image2 = np.random.rand(100, 100, 3)

# Apply mixup
mixed, lam = aug.mixup(image1, image2, alpha=0.2)

print(f"Image 1 shape: {image1.shape}")
print(f"Image 2 shape: {image2.shape}")
print(f"Mixed image shape: {mixed.shape}")
print(f"Mixing coefficient (lambda): {lam:.3f}")
print(f"\nMixed image = {lam:.3f} * image1 + {1-lam:.3f} * image2")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Cutout Augmentation
# ============================================================================
print("EXAMPLE 4: Cutout Augmentation")
print("-" * 80)

aug = ImageAugmenter(seed=42)
image = np.ones((100, 100, 3))

# Apply cutout
cutout_img = aug.cutout(image, n_holes=3, length=20)

zeros_count = (cutout_img == 0).sum()
total_pixels = cutout_img.size

print(f"Original image: all ones")
print(f"After cutout: {zeros_count} zero pixels out of {total_pixels}")
print(f"Percentage masked: {zeros_count / total_pixels * 100:.2f}%")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Random Erasing
# ============================================================================
print("EXAMPLE 5: Random Erasing")
print("-" * 80)

aug = ImageAugmenter(seed=42)
image = np.random.rand(100, 100, 3)

# Apply random erasing with high probability
erased = aug.random_erasing(image, probability=1.0, area_ratio_range=(0.1, 0.3))

print(f"Original image shape: {image.shape}")
print(f"Erased image shape: {erased.shape}")
print(f"Images are different: {not np.allclose(image, erased)}")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Text Augmentation - Synonym Replacement
# ============================================================================
print("EXAMPLE 6: Text Augmentation - Synonym Replacement")
print("-" * 80)

text_aug = TextAugmenter(seed=42)

original_text = "The quick brown fox jumps over the lazy dog"
print(f"Original: {original_text}")

augmented = text_aug.synonym_replacement(original_text, n=2)
print(f"Augmented: {augmented}")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Text Augmentation - Random Operations
# ============================================================================
print("EXAMPLE 7: Text Augmentation - Random Operations")
print("-" * 80)

text_aug = TextAugmenter(seed=42)
text = "Machine learning is transforming the world"

print(f"Original: {text}")

# Random insertion
inserted = text_aug.random_insertion(text, n=2)
print(f"After insertion: {inserted}")

# Random swap
swapped = text_aug.random_swap(text, n=2)
print(f"After swap: {swapped}")

# Random deletion
deleted = text_aug.random_deletion(text, p=0.2)
print(f"After deletion: {deleted}")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Augmentation Pipeline
# ============================================================================
print("EXAMPLE 8: Augmentation Pipeline")
print("-" * 80)

aug = ImageAugmenter(seed=42)

# Create pipeline
pipeline = AugmentationPipeline([
    lambda x: aug.random_rotation(x, max_angle=15),
    lambda x: aug.horizontal_flip(x) if np.random.random() > 0.5 else x,
    lambda x: aug.color_jitter(x, brightness=0.2, contrast=0.2),
    lambda x: aug.gaussian_noise(x, std=0.03)
])

image = np.random.rand(100, 100, 3)
print(f"Original image shape: {image.shape}")

augmented = pipeline(image)
print(f"Augmented image shape: {augmented.shape}")
print(f"Pipeline has {len(pipeline.transforms)} transforms")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Training Data Augmentation
# ============================================================================
print("EXAMPLE 9: Training Data Augmentation")
print("-" * 80)

aug = ImageAugmenter(seed=42)

# Simulate training batch
batch_size = 8
images = [np.random.rand(224, 224, 3) for _ in range(batch_size)]

print(f"Original batch size: {len(images)}")

# Augment each image
augmented_images = []
for img in images:
    # Apply random augmentations
    aug_img = aug.random_rotation(img, max_angle=20)
    aug_img = aug.horizontal_flip(aug_img) if np.random.random() > 0.5 else aug_img
    aug_img = aug.color_jitter(aug_img, brightness=0.2)
    augmented_images.append(aug_img)

print(f"Augmented batch size: {len(augmented_images)}")
print(f"Each image shape: {augmented_images[0].shape}")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Normalization
# ============================================================================
print("EXAMPLE 10: Normalization")
print("-" * 80)

aug = ImageAugmenter()
image = np.random.rand(100, 100, 3)

# ImageNet normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

normalized = aug.normalize(image, mean=mean, std=std)

print(f"Original image range: [{image.min():.3f}, {image.max():.3f}]")
print(f"Normalized image range: [{normalized.min():.3f}, {normalized.max():.3f}]")
print(f"Mean: {mean}")
print(f"Std: {std}")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Data Augmentation for Small Datasets
# ============================================================================
print("EXAMPLE 11: Data Augmentation for Small Datasets")
print("-" * 80)

aug = ImageAugmenter(seed=42)

# Small dataset (10 images)
dataset_size = 10
images = [np.random.rand(100, 100, 3) for _ in range(dataset_size)]

print(f"Original dataset size: {dataset_size}")

# Generate 5 augmented versions per image
augmentation_factor = 5
augmented_dataset = []

for img in images:
    augmented_dataset.append(img)  # Original
    
    for _ in range(augmentation_factor - 1):
        # Random augmentation
        aug_img = aug.random_rotation(img, max_angle=30)
        aug_img = aug.horizontal_flip(aug_img) if np.random.random() > 0.5 else aug_img
        aug_img = aug.color_jitter(aug_img, brightness=0.3, contrast=0.3)
        aug_img = aug.gaussian_noise(aug_img, std=0.05)
        augmented_dataset.append(aug_img)

print(f"Augmented dataset size: {len(augmented_dataset)}")
print(f"Increase factor: {len(augmented_dataset) / dataset_size}x")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Reproducible Augmentation
# ============================================================================
print("EXAMPLE 12: Reproducible Augmentation")
print("-" * 80)

# Same seed = same augmentation
aug1 = ImageAugmenter(seed=42)
aug2 = ImageAugmenter(seed=42)

image = np.random.rand(100, 100, 3)

result1 = aug1.random_rotation(image, max_angle=30)
result2 = aug2.random_rotation(image, max_angle=30)

print(f"Results are identical: {np.allclose(result1, result2)}")

# Different seed = different augmentation
aug3 = ImageAugmenter(seed=123)
result3 = aug3.random_rotation(image, max_angle=30)

print(f"Results are different: {not np.allclose(result1, result3)}")

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: Combining Multiple Augmentations
# ============================================================================
print("EXAMPLE 13: Combining Multiple Augmentations")
print("-" * 80)

aug = ImageAugmenter(seed=42)
image = np.random.rand(224, 224, 3)

print("Applying multiple augmentations in sequence:")

# Step 1: Rotation
image = aug.random_rotation(image, max_angle=15)
print("1. Applied rotation")

# Step 2: Flip
image = aug.horizontal_flip(image)
print("2. Applied horizontal flip")

# Step 3: Crop
image = aug.random_crop(image, crop_size=(200, 200))
print("3. Applied random crop")

# Step 4: Color jitter
image = aug.color_jitter(image, brightness=0.2, contrast=0.2)
print("4. Applied color jitter")

# Step 5: Noise
image = aug.gaussian_noise(image, std=0.03)
print("5. Applied Gaussian noise")

print(f"\nFinal image shape: {image.shape}")

print("\n✓ Example 13 completed\n")

# ============================================================================
# EXAMPLE 14: Text Augmentation for NLP
# ============================================================================
print("EXAMPLE 14: Text Augmentation for NLP")
print("-" * 80)

text_aug = TextAugmenter(seed=42)

# Original sentence
sentence = "Deep learning models require large amounts of training data"
print(f"Original: {sentence}")

# Generate 5 augmented versions
print("\nAugmented versions:")
for i in range(5):
    # Apply random augmentation
    choice = np.random.randint(0, 4)
    
    if choice == 0:
        aug_text = text_aug.synonym_replacement(sentence, n=2)
        method = "Synonym replacement"
    elif choice == 1:
        aug_text = text_aug.random_insertion(sentence, n=1)
        method = "Random insertion"
    elif choice == 2:
        aug_text = text_aug.random_swap(sentence, n=2)
        method = "Random swap"
    else:
        aug_text = text_aug.random_deletion(sentence, p=0.1)
        method = "Random deletion"
    
    print(f"{i+1}. [{method}] {aug_text}")

print("\n✓ Example 14 completed\n")

# ============================================================================
# EXAMPLE 15: Performance Comparison
# ============================================================================
print("EXAMPLE 15: Performance Comparison")
print("-" * 80)

import time

aug = ImageAugmenter(seed=42)
image = np.random.rand(224, 224, 3)

operations = [
    ("Horizontal Flip", lambda: aug.horizontal_flip(image)),
    ("Rotation", lambda: aug.random_rotation(image, max_angle=30)),
    ("Color Jitter", lambda: aug.color_jitter(image)),
    ("Gaussian Noise", lambda: aug.gaussian_noise(image)),
    ("Random Crop", lambda: aug.random_crop(image, (200, 200))),
]

print(f"{'Operation':<20} {'Time (ms)':<15}")
print("-" * 35)

for name, operation in operations:
    start = time.time()
    for _ in range(100):
        operation()
    elapsed = (time.time() - start) * 1000 / 100
    print(f"{name:<20} {elapsed:<15.3f}")

print("\n✓ Example 15 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Basic Image Augmentation")
print("2. ✓ Color Augmentation")
print("3. ✓ Mixup")
print("4. ✓ Cutout")
print("5. ✓ Random Erasing")
print("6. ✓ Text Synonym Replacement")
print("7. ✓ Text Random Operations")
print("8. ✓ Augmentation Pipeline")
print("9. ✓ Training Data Augmentation")
print("10. ✓ Normalization")
print("11. ✓ Small Dataset Augmentation")
print("12. ✓ Reproducible Augmentation")
print("13. ✓ Combining Augmentations")
print("14. ✓ Text Augmentation for NLP")
print("15. ✓ Performance Comparison")
print()
print("You now have a complete understanding of data augmentation!")
print()
print("Next steps:")
print("- Apply augmentation to your datasets")
print("- Experiment with different combinations")
print("- Monitor training/validation performance")
print("- Adjust augmentation strength based on results")
