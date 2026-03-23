"""
Extraction Workflow
-------------------
Repository-level orchestration for Yahoo feature extraction into inspectable
CSV datasets.
"""

from __future__ import annotations

from pathlib import Path

from density_model.config import ExtractConfig
from density_model.data import YahooDailyReturnsLoader, YahooVolatilityPanelBuilder

__all__ = ["extract_features_from_config"]


def extract_features_from_config(config: ExtractConfig) -> Path:
    """
    Extract Yahoo-based raw features and save them to CSV.

    Parameters
    ----------
    config : ExtractConfig
        Typed extraction configuration.

    Returns
    -------
    pathlib.Path
        Output CSV path used by the extraction run.
    """

    loader = YahooDailyReturnsLoader()
    panel_builder = YahooVolatilityPanelBuilder()
    feature_panel = panel_builder.build_feature_panel_from_loader(
        loader=loader,
        request=config.data.to_yahoo_request(),
    )
    output_path = config.output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_panel.to_csv(output_path, index=False)
    return output_path
