# /src/miyagi_machines/__init__.py

from .datasets      import get_processed_dataset
from .custom_ops    import FocalLoss, SupConLoss, GradientReversalLayer
from .models        import Model
from .trainer       import MultiObjectiveTrainer, compute_metrics

# public API
__all__ = [
    "get_processed_dataset",
    "FocalLoss",  # DEPRECATED
    "SupConLoss",
    "GradientReversalLayer",
    "Model",
    "MultiObjectiveTrainer",
    "compute_metrics",
]

