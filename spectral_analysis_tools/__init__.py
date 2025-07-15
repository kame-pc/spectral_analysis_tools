"""Utility functions for spectral analysis."""

from .core import (
    lighten_color,
    gradient_dict,
    PSD,
    PSD_clus,
    CSD,
    CSD_clus,
)
from .generation import generate_autocorr, generate_crosscorr

__all__ = [
    'lighten_color',
    'gradient_dict',
    'PSD',
    'PSD_clus',
    'CSD',
    'CSD_clus',
    'generate_autocorr',
    'generate_crosscorr',
]
