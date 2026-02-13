# New modules to add to ilovetools/ml/__init__.py

# Graph Neural Networks
from .gnn import (
    GCN,
    GAT,
    GraphSAGE,
    GIN,
)

# Autoencoders
from .autoencoder import (
    VanillaAutoencoder,
    DenoisingAutoencoder,
    SparseAutoencoder,
    ContractiveAutoencoder,
    VAE,
)

# Transfer Learning
from .transfer import (
    FeatureExtractor,
    FineTuner,
    GradualUnfreezer,
    DiscriminativeLR,
    DomainAdapter,
    compute_transfer_gap,
    learning_rate_warmup,
)

# Object Detection
from .detection import (
    YOLO,
    FasterRCNN,
    SSD,
    RetinaNet,
    AnchorGenerator,
    compute_iou,
    non_maximum_suppression,
    compute_map,
)
