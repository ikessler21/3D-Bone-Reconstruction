"""
CT scan artifact augmentations for fossil domain adaptation (Phase 1).

RingArtifact          — concentric cylindrical noise bands
MatrixContamination   — random spatial occlusion (rock matrix simulation)
MineralizationInfilling — add random interior points
WeatheringErosion     — delete surface-exposed points
ResolutionDropout     — random subsampling
CTArtifactPipeline    — compose all transforms
"""

# Re-export from augmentations module to avoid duplication
from src.datasets.augmentations import (
    RingArtifact,
    MatrixContamination,
    MineralizationInfilling,
    WeatheringErosion,
    ResolutionDropout,
    CTArtifactAugmentation as CTArtifactPipeline,
)

__all__ = [
    "RingArtifact",
    "MatrixContamination",
    "MineralizationInfilling",
    "WeatheringErosion",
    "ResolutionDropout",
    "CTArtifactPipeline",
]
