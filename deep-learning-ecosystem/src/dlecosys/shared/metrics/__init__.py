"""
Metrics
-------
Scalar metric functions for regression and classification tasks.

All functions accept keyword-only arguments ``y_true`` and ``y_pred`` so they
integrate directly with BaseTrainer's metric computation interface:
``metrics={"rmse": rmse, "acc": accuracy}``.

Functions
---------
mae
    Mean absolute error.
mse
    Mean squared error.
rmse
    Root mean squared error.
mape
    Mean absolute percentage error.
accuracy
    Top-1 accuracy for multi-class classification from logits.
binary_accuracy
    Accuracy for binary classification with configurable sigmoid threshold.
"""

from dlecosys.shared.metrics.regression import mae, mse, rmse, mape
from dlecosys.shared.metrics.classification import accuracy, binary_accuracy

__all__ = [
    "mae",
    "mse",
    "rmse",
    "mape",
    "accuracy",
    "binary_accuracy",
]
