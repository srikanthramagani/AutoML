"""
Training package.
Coordinates model fitting:
  - Trainer: fits preprocessing + model pipelines during HPO and CV.
  - FinalTrainer: retrains the best pipeline on the full training set.
"""
from .trainer import Trainer, TrainingResult
from .final_training import FinalTrainer

__all__ = ["Trainer", "TrainingResult", "FinalTrainer"]
