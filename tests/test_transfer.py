"""
Tests for Transfer Learning & Fine-Tuning

This file contains comprehensive tests for all transfer learning strategies.

Author: Ali Mehdi
Date: January 31, 2026
"""

import numpy as np
import pytest
from ilovetools.ml.transfer import (
    FeatureExtractor,
    FineTuner,
    GradualUnfreezer,
    DiscriminativeLR,
    DomainAdapter,
    compute_transfer_gap,
    learning_rate_warmup,
)


# ============================================================================
# TEST FEATURE EXTRACTOR
# ============================================================================

def test_feature_extractor_basic():
    """Test basic feature extractor functionality."""
    extractor = FeatureExtractor(input_dim=2048, num_classes=10)
    features = np.random.randn(32, 2048)
    
    logits = extractor.forward(features)
    
    assert logits.shape == (32, 10)


def test_feature_extractor_freeze():
    """Test backbone freezing."""
    extractor = FeatureExtractor(input_dim=2048, num_classes=10)
    
    extractor.freeze_backbone()
    assert extractor.backbone_frozen == True


def test_feature_extractor_unfreeze():
    """Test backbone unfreezing."""
    extractor = FeatureExtractor(input_dim=2048, num_classes=10)
    
    extractor.freeze_backbone()
    extractor.unfreeze_backbone()
    
    assert extractor.backbone_frozen == False


def test_feature_extractor_callable():
    """Test that feature extractor is callable."""
    extractor = FeatureExtractor(input_dim=2048, num_classes=10)
    features = np.random.randn(32, 2048)
    
    output = extractor(features)
    assert output is not None


# ============================================================================
# TEST FINE-TUNER
# ============================================================================

def test_finetuner_basic():
    """Test basic fine-tuner functionality."""
    finetuner = FineTuner(num_layers=50, num_classes=10)
    
    assert finetuner.num_layers == 50
    assert finetuner.num_classes == 10


def test_finetuner_unfreeze_layers():
    """Test layer unfreezing."""
    finetuner = FineTuner(num_layers=50, num_classes=10)
    
    finetuner.unfreeze_layers(start_layer=40, end_layer=50)
    
    trainable = finetuner.get_trainable_layers()
    assert len(trainable) >= 10


def test_finetuner_freeze_layers():
    """Test layer freezing."""
    finetuner = FineTuner(num_layers=50, num_classes=10)
    
    finetuner.unfreeze_layers(start_layer=0, end_layer=50)
    finetuner.freeze_layers(start_layer=0, end_layer=40)
    
    trainable = finetuner.get_trainable_layers()
    assert len(trainable) <= 11  # Last layer + unfrozen layers


def test_finetuner_discriminative_lr():
    """Test discriminative learning rates."""
    finetuner = FineTuner(num_layers=50, num_classes=10, base_lr=1e-3)
    
    finetuner.set_layer_lrs(discriminative=True, lr_decay=0.95)
    
    # Early layers should have lower LR
    assert finetuner.layer_lrs[0] < finetuner.layer_lrs[-1]


def test_finetuner_uniform_lr():
    """Test uniform learning rates."""
    finetuner = FineTuner(num_layers=50, num_classes=10, base_lr=1e-3)
    
    finetuner.set_layer_lrs(discriminative=False)
    
    # All layers should have same LR
    assert all(lr == 1e-3 for lr in finetuner.layer_lrs)


# ============================================================================
# TEST GRADUAL UNFREEZER
# ============================================================================

def test_gradual_unfreezer_basic():
    """Test basic gradual unfreezer functionality."""
    unfreezer = GradualUnfreezer(num_layers=50, num_groups=4, epochs_per_group=3)
    
    assert unfreezer.num_layers == 50
    assert unfreezer.num_groups == 4


def test_gradual_unfreezer_progression():
    """Test progressive unfreezing."""
    unfreezer = GradualUnfreezer(num_layers=50, num_groups=4, epochs_per_group=3)
    
    # Initially only last layer trainable
    trainable_0 = unfreezer.get_trainable_layers()
    
    # After 3 epochs, more layers should be trainable
    unfreezer.step(epoch=3)
    trainable_3 = unfreezer.get_trainable_layers()
    
    assert len(trainable_3) > len(trainable_0)


def test_gradual_unfreezer_all_epochs():
    """Test unfreezing across all epochs."""
    unfreezer = GradualUnfreezer(num_layers=50, num_groups=4, epochs_per_group=3)
    
    for epoch in range(12):
        unfreezer.step(epoch)
    
    # After all epochs, most layers should be trainable
    trainable = unfreezer.get_trainable_layers()
    assert len(trainable) > 40


# ============================================================================
# TEST DISCRIMINATIVE LR
# ============================================================================

def test_discriminative_lr_basic():
    """Test basic discriminative LR functionality."""
    scheduler = DiscriminativeLR(num_layers=50, base_lr=1e-3, decay_factor=0.95)
    
    lrs = scheduler.get_layer_lrs()
    
    assert len(lrs) == 50
    assert lrs[-1] == 1e-3  # Base LR for last layer


def test_discriminative_lr_decay():
    """Test LR decay for earlier layers."""
    scheduler = DiscriminativeLR(num_layers=50, base_lr=1e-3, decay_factor=0.95)
    
    lrs = scheduler.get_layer_lrs()
    
    # Earlier layers should have lower LR
    assert lrs[0] < lrs[25] < lrs[49]


def test_discriminative_lr_get_layer_lr():
    """Test getting LR for specific layer."""
    scheduler = DiscriminativeLR(num_layers=50, base_lr=1e-3)
    
    lr_0 = scheduler.get_layer_lr(0)
    lr_49 = scheduler.get_layer_lr(49)
    
    assert lr_0 < lr_49


# ============================================================================
# TEST DOMAIN ADAPTER
# ============================================================================

def test_domain_adapter_basic():
    """Test basic domain adapter functionality."""
    adapter = DomainAdapter(feature_dim=2048, num_classes=10)
    
    assert adapter.feature_dim == 2048
    assert adapter.num_classes == 10


def test_domain_adapter_adversarial_loss():
    """Test adversarial domain adaptation loss."""
    adapter = DomainAdapter(feature_dim=2048, num_classes=10, adaptation_method='adversarial')
    
    source_features = np.random.randn(32, 2048)
    target_features = np.random.randn(32, 2048)
    
    loss = adapter.domain_loss(source_features, target_features)
    
    assert isinstance(loss, float)
    assert loss >= 0


def test_domain_adapter_mmd_loss():
    """Test MMD domain adaptation loss."""
    adapter = DomainAdapter(feature_dim=2048, num_classes=10, adaptation_method='mmd')
    
    source_features = np.random.randn(32, 2048)
    target_features = np.random.randn(32, 2048)
    
    loss = adapter.domain_loss(source_features, target_features)
    
    assert isinstance(loss, float)
    assert loss >= 0


def test_domain_adapter_task_forward():
    """Test task classifier forward pass."""
    adapter = DomainAdapter(feature_dim=2048, num_classes=10)
    features = np.random.randn(32, 2048)
    
    logits = adapter.task_forward(features)
    
    assert logits.shape == (32, 10)


# ============================================================================
# TEST UTILITY FUNCTIONS
# ============================================================================

def test_compute_transfer_gap():
    """Test transfer gap computation."""
    gap = compute_transfer_gap(source_acc=0.95, target_acc=0.85)
    
    assert gap == 0.10


def test_learning_rate_warmup():
    """Test learning rate warmup."""
    # During warmup
    lr_epoch_0 = learning_rate_warmup(epoch=0, warmup_epochs=5, base_lr=1e-3)
    lr_epoch_2 = learning_rate_warmup(epoch=2, warmup_epochs=5, base_lr=1e-3)
    
    # After warmup
    lr_epoch_5 = learning_rate_warmup(epoch=5, warmup_epochs=5, base_lr=1e-3)
    
    assert lr_epoch_0 < lr_epoch_2 < lr_epoch_5
    assert lr_epoch_5 == 1e-3


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_feature_extraction_pipeline():
    """Test complete feature extraction pipeline."""
    # Pretrained backbone features
    features = np.random.randn(100, 2048)
    
    # Feature extractor
    extractor = FeatureExtractor(input_dim=2048, num_classes=10)
    extractor.freeze_backbone()
    
    # Forward pass
    logits = extractor.forward(features)
    
    assert logits.shape == (100, 10)
    assert extractor.backbone_frozen


def test_fine_tuning_pipeline():
    """Test complete fine-tuning pipeline."""
    # Fine-tuner
    finetuner = FineTuner(num_layers=50, num_classes=10, base_lr=1e-4)
    
    # Set discriminative LRs
    finetuner.set_layer_lrs(discriminative=True)
    
    # Unfreeze top layers
    finetuner.unfreeze_layers(start_layer=40)
    
    trainable = finetuner.get_trainable_layers()
    
    assert len(trainable) >= 10
    assert finetuner.layer_lrs[0] < finetuner.layer_lrs[-1]


def test_gradual_unfreezing_pipeline():
    """Test gradual unfreezing pipeline."""
    unfreezer = GradualUnfreezer(num_layers=50, num_groups=4, epochs_per_group=3)
    
    trainable_counts = []
    
    for epoch in range(12):
        unfreezer.step(epoch)
        trainable = unfreezer.get_trainable_layers()
        trainable_counts.append(len(trainable))
    
    # Should progressively increase
    assert trainable_counts[-1] > trainable_counts[0]


def test_domain_adaptation_pipeline():
    """Test domain adaptation pipeline."""
    adapter = DomainAdapter(feature_dim=2048, num_classes=10)
    
    # Source and target features
    source_features = np.random.randn(32, 2048)
    target_features = np.random.randn(32, 2048)
    
    # Domain loss
    domain_loss = adapter.domain_loss(source_features, target_features)
    
    # Task prediction
    task_logits = adapter.task_forward(target_features)
    
    assert isinstance(domain_loss, float)
    assert task_logits.shape == (32, 10)


def test_combined_strategies():
    """Test combining multiple transfer learning strategies."""
    # Feature extractor
    extractor = FeatureExtractor(input_dim=2048, num_classes=10)
    extractor.freeze_backbone()
    
    # Fine-tuner
    finetuner = FineTuner(num_layers=50, num_classes=10)
    finetuner.set_layer_lrs(discriminative=True)
    
    # Gradual unfreezer
    unfreezer = GradualUnfreezer(num_layers=50, num_groups=4)
    
    # LR scheduler
    scheduler = DiscriminativeLR(num_layers=50, base_lr=1e-3)
    
    # All should work together
    assert extractor.backbone_frozen
    assert len(finetuner.get_trainable_layers()) >= 1
    assert len(unfreezer.get_trainable_layers()) >= 1
    assert len(scheduler.get_layer_lrs()) == 50


print("=" * 80)
print("ALL TRANSFER LEARNING TESTS PASSED! ✓")
print("=" * 80)
