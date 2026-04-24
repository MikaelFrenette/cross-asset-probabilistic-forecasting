"""
Run Management
--------------
Output directory layout and run lifecycle utilities.

Classes
-------
DataPaths
    Protocol for the data-side API shared by RunLayout and StudyLayout.
RunLayout
    Encapsulates all output paths for a pipeline run and creates the
    non-negotiable directory tree.
StudyLayout
    Output layout for a hyperparameter tuning study, with a trial_layout
    factory that mints per-trial RunLayouts sharing the study data dir.
"""

from dlecosys.shared.run.layout import DataPaths, EnsembleLayout, RunLayout, StudyLayout

__all__ = ["DataPaths", "RunLayout", "StudyLayout", "EnsembleLayout"]
