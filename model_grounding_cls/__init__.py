"""
ProteinSAM: Segment Anything Model for protein functional region grounding.

This module provides:
- ProteinSAM: Main model class
- ProteinEncoder: ESM2-based protein encoder
- PromptEncoder: Text and position prompt encoder  
- PositionDecoder: Lightweight decoder for position prediction
- ProteinSAMDataset: Dataset loader
- ProteinSAMCollator: Data collator for batching
"""

from .protein_sam import ProteinSAM
from .protein_encoder import ProteinEncoder
from .prompt_encoder import PromptEncoder
from .position_decoder import PositionDecoder
from .dataset import ProteinSAMDataset, ProteinSAMCollator, get_datasets_and_collator

__version__ = "1.0.0"

__all__ = [
    "ProteinSAM",
    "ProteinEncoder", 
    "PromptEncoder",
    "PositionDecoder",
    "ProteinSAMDataset",
    "ProteinSAMCollator",
    "get_datasets_and_collator"
]