"""
Ensembling
----------
Bagging ensembles built on top of the standard training lane. Each estimator
is an independent fit via ``run_training`` with a unique bootstrap sample as
its train set and the complementary out-of-bag (OOB) samples as its
validation set — so EarlyStopping and ModelCheckpoint target OOB performance
with no additional machinery.

Import directly from submodules. Optuna is not required.

Submodules
----------
base
    BaseEnsemble parent contract.
bagging
    BaggingEnsemble (sample bootstrap).
aggregation
    Prediction aggregation functions.
build
    build_ensemble factory.
runner
    run_ensemble entrypoint + console rendering.
"""

__all__: list[str] = []
