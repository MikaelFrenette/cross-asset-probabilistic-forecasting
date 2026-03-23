"""
Training Package
----------------
Public training abstractions, callbacks, and supervised trainer implementations.
"""

from density_model.training.base import BaseTrainer
from density_model.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    ModelCheckpointConfig,
)
from density_model.training.losses import (
    GaussianLogLikelihood,
    MaskedGaussianLogLikelihood,
    MaskedMAELoss,
    MaskedMSELoss,
)
from density_model.training.supervised import SupervisedTrainer, SupervisedTrainerConfig

__all__ = [
    "BaseTrainer",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "GaussianLogLikelihood",
    "MaskedGaussianLogLikelihood",
    "ModelCheckpoint",
    "ModelCheckpointConfig",
    "MaskedMAELoss",
    "MaskedMSELoss",
    "SupervisedTrainer",
    "SupervisedTrainerConfig",
]
