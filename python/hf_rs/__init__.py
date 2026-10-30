# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""hf-rs native calculation API and optional ASE integration."""

from importlib.metadata import PackageNotFoundError, version

from . import core as _core
from .ase_calc import HFCalculator, HFRSCalculator
from .core import __all__ as _core_exports
from .core import _rust

for _name in _core_exports:
    globals()[_name] = getattr(_core, _name)

try:
    __version__ = _rust.__version__
except AttributeError:
    try:
        __version__ = version("hf-rs-python")
    except PackageNotFoundError:
        __version__ = "0+unknown"

__author__ = "ss0832"
__license__ = "Apache-2.0"
__all__ = _core_exports + ["HFCalculator", "HFRSCalculator"]
