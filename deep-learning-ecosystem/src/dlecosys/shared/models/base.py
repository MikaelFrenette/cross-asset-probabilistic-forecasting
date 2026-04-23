"""
Model Convention Base
---------------------
Base configuration class that all registered model configs must subclass.

Classes
-------
ModelConfig
    Pydantic base model enforcing the dlecosys model convention.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["ModelConfig"]


class ModelConfig(BaseModel):
    """
    Base configuration for all registered models.

    Every model registered with :func:`register` must follow this convention:

    1. Define a companion config class that subclasses ``ModelConfig``.
    2. Declare a class attribute ``config_class`` pointing to that config class.
    3. Implement ``from_config(cls, config) -> nn.Module``.

    Example
    -------
    ::

        class MyModelConfig(ModelConfig):
            input_dim: int
            hidden_dim: int = 64

        @register("my_model")
        class MyModel(nn.Module):
            config_class = MyModelConfig

            @classmethod
            def from_config(cls, config: MyModelConfig) -> "MyModel":
                return cls(config)
    """

    model_config = ConfigDict(frozen=True)
