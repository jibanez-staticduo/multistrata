"""Opt-in Strata runtime. The image supplies CUDA, never a vLLM process."""

from .runtime import StrataNativeRuntime, StrataReferenceRuntime

SPARKRUN_PLUGIN_API_VERSION = 1
__version__ = "0.1.0"

__all__ = ["StrataNativeRuntime", "StrataReferenceRuntime"]
