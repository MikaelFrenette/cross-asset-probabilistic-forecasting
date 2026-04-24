"""
Inference
---------
Batched inference pipeline for trained PyTorch models.

Classes
-------
Predictor
    Wraps a trained model with optional preprocessing and runs batched
    forward passes at inference time. Loads directly from ModelCheckpoint
    files via ``Predictor.from_checkpoint``.
"""

from dlecosys.shared.inference.predictor import Predictor

__all__ = ["Predictor"]
