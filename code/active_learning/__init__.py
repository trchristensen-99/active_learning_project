"""
Active Learning Framework for Genomic Sequence Design

This package provides a modular framework for iterative active learning
in genomic sequence analysis, following the PIONEER approach.
"""

from .oracle import BaseOracle, EnsembleOracle
from .student import DeepSTARRStudent
from .proposal import BaseProposalStrategy, RandomProposalStrategy, MixedProposalStrategy
from .acquisition import BaseAcquisitionFunction, RandomAcquisition, UncertaintyAcquisition
from .trainer import BaseActiveLearningTrainer, DeepSTARRActiveLearningTrainer
from .cycle import ActiveLearningCycle
from .config_manager import ConfigurationManager
from .checkpoint import CheckpointManager
from .oracle_paths import get_oracle_path, parse_oracle_path

__all__ = [
    "BaseOracle",
    "EnsembleOracle", 
    "DeepSTARRStudent",
    "BaseProposalStrategy",
    "RandomProposalStrategy",
    "MixedProposalStrategy",
    "BaseAcquisitionFunction",
    "RandomAcquisition",
    "UncertaintyAcquisition",
    "BaseActiveLearningTrainer",
    "DeepSTARRActiveLearningTrainer",
    "ActiveLearningCycle",
    "ConfigurationManager",
    "CheckpointManager",
    "get_oracle_path",
    "parse_oracle_path"
]


