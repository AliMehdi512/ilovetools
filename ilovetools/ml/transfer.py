"""
Transfer Learning & Fine-Tuning Suite

This module implements transfer learning and fine-tuning strategies for leveraging
pretrained models. Transfer learning reuses knowledge from source tasks to improve
performance on target tasks with limited data.

Implemented Strategies:
1. FeatureExtractor - Freeze backbone, train classifier only
2. FineTuner - Unfreeze and retrain with small learning rates
3. GradualUnfreezer - Progressive layer unfreezing
4. DiscriminativeLR - Layer-wise learning rates
5. DomainAdapter - Adapt pretrained models to new domains

Key Benefits:
- Leverage pretrained models (ImageNet, BERT, GPT)
- Reduce training time and data requirements
- Better performance with limited labeled data
- Avoid training from scratch
- Domain adaptation capabilities

References:
- Transfer Learning: Pan & Yang, "A Survey on Transfer Learning" (2010)
- Fine-Tuning: Yosinski et al., "How transferable are features in deep neural networks?" (2014)
- Discriminative Fine-Tuning: Howard & Ruder, "Universal Language Model Fine-tuning" (2018)
- Domain Adaptation: Ganin et al., "Domain-Adversarial Training" (2016)

Author: Ali Mehdi
Date: January 31, 2026
"""

import numpy as np
from typing import Optional, List, Tuple, Dict


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """
    Feature Extraction for Transfer Learning.
    
    Freezes pretrained backbone and trains only the classifier head. Ideal for
    small datasets or when source and target domains are similar.
    
    Strategy:
        1. Load pretrained model (e.g., ResNet on ImageNet)
        2. Freeze all backbone layers
        3. Replace classifier head
        4. Train only the head on target data
    
    Args:
        input_dim: Input feature dimension from backbone
        num_classes: Number of target classes
        hidden_dims: Hidden layer dimensions for classifier (default: [512])
        pretrained_name: Name of pretrained model (for reference)
    
    Example:
        >>> extractor = FeatureExtractor(input_dim=2048, num_classes=10)
        >>> extractor.freeze_backbone()
        >>> features = np.random.randn(32, 2048)  # From frozen backbone
        >>> logits = extractor.forward(features)
        >>> print(logits.shape)  # (32, 10)
    
    Use Case:
        Small datasets, similar domains, fast training, low compute
    
    Reference:
        Razavian et al., "CNN Features off-the-shelf" (2014)
    """
    
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: Optional[List[int]] = None,
                 pretrained_name: str = 'resnet50'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims or [512]
        self.pretrained_name = pretrained_name
        self.backbone_frozen = False
        
        # Initialize classifier head
        self.classifier_weights = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            w = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(hidden_dim)
            self.classifier_weights.append((w, b))
            prev_dim = hidden_dim
        
        # Output layer
        w = np.random.randn(prev_dim, num_classes) * np.sqrt(2.0 / prev_dim)
        b = np.zeros(num_classes)
        self.classifier_weights.append((w, b))
    
    def freeze_backbone(self):
        """Freeze backbone layers (no gradient updates)."""
        self.backbone_frozen = True
        print(f"✓ Backbone frozen ({self.pretrained_name})")
        print("  Only classifier head will be trained")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone layers (allow gradient updates)."""
        self.backbone_frozen = False
        print(f"✓ Backbone unfrozen ({self.pretrained_name})")
        print("  All layers will be trained")
    
    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass through classifier head.
        
        Args:
            features: Features from frozen backbone (batch_size, input_dim)
        
        Returns:
            Class logits (batch_size, num_classes)
        """
        h = features
        
        for i, (w, b) in enumerate(self.classifier_weights):
            h = h @ w + b
            
            # ReLU activation (except last layer)
            if i < len(self.classifier_weights) - 1:
                h = np.maximum(0, h)
        
        return h
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        return self.forward(features)


# ============================================================================
# FINE-TUNER
# ============================================================================

class FineTuner:
    """
    Fine-Tuning for Transfer Learning.
    
    Unfreezes pretrained layers and retrains with small learning rates. Better
    adaptation to target domain than feature extraction.
    
    Strategy:
        1. Load pretrained model
        2. Replace output layer
        3. Unfreeze some/all layers
        4. Train with small learning rates (1e-5 to 1e-4)
        5. Use discriminative learning rates (lower for early layers)
    
    Args:
        num_layers: Total number of layers in model
        num_classes: Number of target classes
        base_lr: Base learning rate for fine-tuning (default: 1e-4)
        freeze_until_layer: Freeze layers before this index (default: None)
    
    Example:
        >>> finetuner = FineTuner(num_layers=50, num_classes=10, base_lr=1e-4)
        >>> finetuner.set_layer_lrs(discriminative=True)
        >>> finetuner.unfreeze_layers(start_layer=40)
        >>> print(finetuner.get_trainable_layers())
    
    Use Case:
        Domain shift, larger datasets, higher accuracy needed
    
    Reference:
        Yosinski et al., "How transferable are features" (2014)
    """
    
    def __init__(self, num_layers: int, num_classes: int,
                 base_lr: float = 1e-4,
                 freeze_until_layer: Optional[int] = None):
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.base_lr = base_lr
        self.freeze_until_layer = freeze_until_layer
        
        # Layer states (frozen or trainable)
        self.layer_frozen = [True] * num_layers
        
        # Learning rates per layer
        self.layer_lrs = [base_lr] * num_layers
        
        # Unfreeze output layer by default
        self.layer_frozen[-1] = False
    
    def unfreeze_layers(self, start_layer: int = 0, end_layer: Optional[int] = None):
        """
        Unfreeze layers in range [start_layer, end_layer).
        
        Args:
            start_layer: First layer to unfreeze
            end_layer: Last layer to unfreeze (exclusive)
        """
        end_layer = end_layer or self.num_layers
        
        for i in range(start_layer, end_layer):
            self.layer_frozen[i] = False
        
        num_unfrozen = sum(not frozen for frozen in self.layer_frozen)
        print(f"✓ Unfrozen layers {start_layer} to {end_layer-1}")
        print(f"  Trainable layers: {num_unfrozen}/{self.num_layers}")
    
    def freeze_layers(self, start_layer: int = 0, end_layer: Optional[int] = None):
        """
        Freeze layers in range [start_layer, end_layer).
        
        Args:
            start_layer: First layer to freeze
            end_layer: Last layer to freeze (exclusive)
        """
        end_layer = end_layer or self.num_layers
        
        for i in range(start_layer, end_layer):
            self.layer_frozen[i] = True
        
        num_frozen = sum(self.layer_frozen)
        print(f"✓ Frozen layers {start_layer} to {end_layer-1}")
        print(f"  Frozen layers: {num_frozen}/{self.num_layers}")
    
    def set_layer_lrs(self, discriminative: bool = True, lr_decay: float = 0.95):
        """
        Set layer-wise learning rates.
        
        Args:
            discriminative: Use discriminative learning rates (lower for early layers)
            lr_decay: Decay factor for earlier layers (default: 0.95)
        """
        if discriminative:
            # Discriminative learning rates: lower for early layers
            for i in range(self.num_layers):
                # Exponential decay from output to input
                decay_factor = lr_decay ** (self.num_layers - i - 1)
                self.layer_lrs[i] = self.base_lr * decay_factor
            
            print(f"✓ Discriminative learning rates set")
            print(f"  Layer 0 LR: {self.layer_lrs[0]:.2e}")
            print(f"  Layer {self.num_layers-1} LR: {self.layer_lrs[-1]:.2e}")
        else:
            # Uniform learning rate
            self.layer_lrs = [self.base_lr] * self.num_layers
            print(f"✓ Uniform learning rate: {self.base_lr:.2e}")
    
    def get_trainable_layers(self) -> List[int]:
        """Get indices of trainable layers."""
        return [i for i, frozen in enumerate(self.layer_frozen) if not frozen]
    
    def get_layer_lr(self, layer_idx: int) -> float:
        """Get learning rate for specific layer."""
        return self.layer_lrs[layer_idx] if not self.layer_frozen[layer_idx] else 0.0


# ============================================================================
# GRADUAL UNFREEZER
# ============================================================================

class GradualUnfreezer:
    """
    Gradual Unfreezing Strategy.
    
    Progressively unfreezes layers from top to bottom during training. Prevents
    catastrophic forgetting and allows better adaptation.
    
    Strategy:
        1. Start with all layers frozen except classifier
        2. Train for N epochs
        3. Unfreeze top layer group
        4. Train for N epochs
        5. Repeat until all layers unfrozen
    
    Args:
        num_layers: Total number of layers
        num_groups: Number of layer groups to unfreeze progressively (default: 4)
        epochs_per_group: Epochs to train before unfreezing next group (default: 3)
    
    Example:
        >>> unfreezer = GradualUnfreezer(num_layers=50, num_groups=4)
        >>> for epoch in range(12):
        ...     unfreezer.step(epoch)
        ...     trainable = unfreezer.get_trainable_layers()
        ...     print(f"Epoch {epoch}: {len(trainable)} trainable layers")
    
    Use Case:
        Large domain shift, prevent catastrophic forgetting, stable training
    
    Reference:
        Howard & Ruder, "Universal Language Model Fine-tuning" (2018)
    """
    
    def __init__(self, num_layers: int, num_groups: int = 4,
                 epochs_per_group: int = 3):
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.epochs_per_group = epochs_per_group
        
        # Calculate layers per group
        self.layers_per_group = num_layers // num_groups
        
        # All layers frozen initially except last
        self.layer_frozen = [True] * num_layers
        self.layer_frozen[-1] = False  # Classifier always trainable
        
        self.current_group = 0
    
    def step(self, epoch: int):
        """
        Update frozen layers based on current epoch.
        
        Args:
            epoch: Current training epoch
        """
        # Determine which group to unfreeze
        target_group = min(epoch // self.epochs_per_group, self.num_groups - 1)
        
        if target_group > self.current_group:
            # Unfreeze next group
            self.unfreeze_group(target_group)
            self.current_group = target_group
    
    def unfreeze_group(self, group_idx: int):
        """
        Unfreeze a specific layer group.
        
        Args:
            group_idx: Index of group to unfreeze (0 = top layers)
        """
        # Calculate layer range for this group (from top)
        start_layer = self.num_layers - (group_idx + 1) * self.layers_per_group
        end_layer = self.num_layers - group_idx * self.layers_per_group
        
        start_layer = max(0, start_layer)
        
        for i in range(start_layer, end_layer):
            self.layer_frozen[i] = False
        
        num_trainable = sum(not frozen for frozen in self.layer_frozen)
        print(f"✓ Unfroze group {group_idx} (layers {start_layer}-{end_layer-1})")
        print(f"  Trainable layers: {num_trainable}/{self.num_layers}")
    
    def get_trainable_layers(self) -> List[int]:
        """Get indices of currently trainable layers."""
        return [i for i, frozen in enumerate(self.layer_frozen) if not frozen]


# ============================================================================
# DISCRIMINATIVE LEARNING RATE SCHEDULER
# ============================================================================

class DiscriminativeLR:
    """
    Discriminative Learning Rate Scheduler.
    
    Assigns different learning rates to different layers. Lower rates for early
    layers (general features) and higher rates for later layers (task-specific).
    
    Formula:
        lr_layer_i = base_lr * decay^(num_layers - i - 1)
    
    Args:
        num_layers: Total number of layers
        base_lr: Base learning rate for output layer (default: 1e-3)
        decay_factor: Decay factor for earlier layers (default: 0.95)
    
    Example:
        >>> scheduler = DiscriminativeLR(num_layers=50, base_lr=1e-3, decay_factor=0.95)
        >>> lrs = scheduler.get_layer_lrs()
        >>> print(f"Layer 0 LR: {lrs[0]:.2e}")
        >>> print(f"Layer 49 LR: {lrs[49]:.2e}")
    
    Use Case:
        Fine-tuning deep networks, preserve early features, adapt late features
    
    Reference:
        Howard & Ruder, "Universal Language Model Fine-tuning" (2018)
    """
    
    def __init__(self, num_layers: int, base_lr: float = 1e-3,
                 decay_factor: float = 0.95):
        self.num_layers = num_layers
        self.base_lr = base_lr
        self.decay_factor = decay_factor
        
        # Compute layer-wise learning rates
        self.layer_lrs = self._compute_layer_lrs()
    
    def _compute_layer_lrs(self) -> List[float]:
        """Compute learning rate for each layer."""
        lrs = []
        
        for i in range(self.num_layers):
            # Exponential decay from output to input
            decay = self.decay_factor ** (self.num_layers - i - 1)
            lr = self.base_lr * decay
            lrs.append(lr)
        
        return lrs
    
    def get_layer_lrs(self) -> List[float]:
        """Get learning rates for all layers."""
        return self.layer_lrs
    
    def get_layer_lr(self, layer_idx: int) -> float:
        """Get learning rate for specific layer."""
        return self.layer_lrs[layer_idx]
    
    def print_summary(self):
        """Print learning rate summary."""
        print("Discriminative Learning Rates:")
        print(f"  Base LR: {self.base_lr:.2e}")
        print(f"  Decay factor: {self.decay_factor}")
        print(f"  Layer 0 LR: {self.layer_lrs[0]:.2e}")
        print(f"  Layer {self.num_layers//2} LR: {self.layer_lrs[self.num_layers//2]:.2e}")
        print(f"  Layer {self.num_layers-1} LR: {self.layer_lrs[-1]:.2e}")


# ============================================================================
# DOMAIN ADAPTER
# ============================================================================

class DomainAdapter:
    """
    Domain Adaptation for Transfer Learning.
    
    Adapts pretrained models to new domains with distribution shift. Uses
    techniques like domain-adversarial training and feature alignment.
    
    Strategy:
        1. Load pretrained model (source domain)
        2. Add domain classifier
        3. Train to minimize task loss and maximize domain confusion
        4. Align source and target feature distributions
    
    Args:
        feature_dim: Feature dimension from backbone
        num_classes: Number of target classes
        adaptation_method: Adaptation method ('adversarial', 'mmd') (default: 'adversarial')
    
    Example:
        >>> adapter = DomainAdapter(feature_dim=2048, num_classes=10)
        >>> source_features = np.random.randn(32, 2048)
        >>> target_features = np.random.randn(32, 2048)
        >>> loss = adapter.domain_loss(source_features, target_features)
        >>> print(f"Domain adaptation loss: {loss:.4f}")
    
    Use Case:
        Domain shift (medical → natural images), cross-domain transfer
    
    Reference:
        Ganin et al., "Domain-Adversarial Training" (2016)
    """
    
    def __init__(self, feature_dim: int, num_classes: int,
                 adaptation_method: str = 'adversarial'):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.adaptation_method = adaptation_method
        
        # Task classifier
        self.task_classifier = np.random.randn(feature_dim, num_classes) * 0.01
        
        # Domain classifier (binary: source vs target)
        self.domain_classifier = np.random.randn(feature_dim, 2) * 0.01
    
    def domain_loss(self, source_features: np.ndarray, 
                    target_features: np.ndarray) -> float:
        """
        Compute domain adaptation loss.
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
        
        Returns:
            Domain adaptation loss
        """
        if self.adaptation_method == 'adversarial':
            # Domain-adversarial loss
            # Predict domain labels
            source_domain_logits = source_features @ self.domain_classifier
            target_domain_logits = target_features @ self.domain_classifier
            
            # Source should be classified as domain 0
            source_loss = -np.mean(np.log(self._softmax(source_domain_logits)[:, 0] + 1e-8))
            
            # Target should be classified as domain 1
            target_loss = -np.mean(np.log(self._softmax(target_domain_logits)[:, 1] + 1e-8))
            
            return source_loss + target_loss
        
        elif self.adaptation_method == 'mmd':
            # Maximum Mean Discrepancy
            source_mean = np.mean(source_features, axis=0)
            target_mean = np.mean(target_features, axis=0)
            
            mmd = np.sum((source_mean - target_mean) ** 2)
            return mmd
        
        return 0.0
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def task_forward(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through task classifier."""
        return features @ self.task_classifier


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_transfer_gap(source_acc: float, target_acc: float) -> float:
    """
    Compute transfer learning gap.
    
    Args:
        source_acc: Accuracy on source domain
        target_acc: Accuracy on target domain
    
    Returns:
        Transfer gap (lower is better)
    
    Example:
        >>> gap = compute_transfer_gap(source_acc=0.95, target_acc=0.85)
        >>> print(f"Transfer gap: {gap:.2%}")
    """
    return source_acc - target_acc


def learning_rate_warmup(epoch: int, warmup_epochs: int, base_lr: float) -> float:
    """
    Learning rate warmup schedule.
    
    Args:
        epoch: Current epoch
        warmup_epochs: Number of warmup epochs
        base_lr: Base learning rate after warmup
    
    Returns:
        Learning rate for current epoch
    
    Example:
        >>> for epoch in range(10):
        ...     lr = learning_rate_warmup(epoch, warmup_epochs=5, base_lr=1e-3)
        ...     print(f"Epoch {epoch}: LR = {lr:.2e}")
    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


__all__ = [
    'FeatureExtractor',
    'FineTuner',
    'GradualUnfreezer',
    'DiscriminativeLR',
    'DomainAdapter',
    'compute_transfer_gap',
    'learning_rate_warmup',
]
