"""
Acquisition strategies for active learning.

Provides various strategies for selecting which samples to acquire (label) next
from the unlabeled pool.
"""

from .strategies import (
    AcquisitionStrategy,
    RandomAcquisition,
    UncertaintyAcquisition,
    DiversityAcquisition,
    HybridAcquisition
)

__all__ = [
    'AcquisitionStrategy',
    'RandomAcquisition',
    'UncertaintyAcquisition',
    'DiversityAcquisition',
    'HybridAcquisition',
]
