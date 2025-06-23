# /src/miyagi_machines/__init__.py

from .models.model  import Model
from .datasets.dataset  import get_processed_dataset
from .trainer       import MultiObjectiveTrainer, compute_metrics

# public API
__all__ = [
    "get_processed_dataset",
    "Model",
    "MultiObjectiveTrainer",
    "compute_metrics",
]

