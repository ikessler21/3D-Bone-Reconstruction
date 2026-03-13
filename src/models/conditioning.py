"""
Conditioning encoders for RELIC.

PhyloEmbedding   — Poincaré ball embedding for taxonomic identity
MorphoBERT       — BioBERT encoder for morphological text descriptions
CLIPImageEncoder — CLIP vision encoder for optional reference images
TaxonomyEncoder  — combined conditioning encoder (phylo + morpho + image)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PhyloEmbedding
# ---------------------------------------------------------------------------

class PhyloEmbedding(nn.Module):
    """
    Poincaré ball embedding table for taxonomic identity.

    Maps integer taxon IDs to fixed-dimensional hyperbolic vectors.
    The embedding is trained separately (see src/utils/phylo.py) and
    frozen during RELIC training.

    Parameters
    ----------
    n_taxa    : int    vocabulary size (number of unique taxa)
    dim       : int    embedding dimension (default 64)
    taxa_list : list   ordered list of taxon name strings (for lookup)
    curvature : float  Poincaré ball curvature parameter
    """

    def __init__(
        self,
        n_taxa: int,
        dim: int = 64,
        taxa_list: Optional[List[str]] = None,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_taxa = n_taxa
        self.taxa_list = taxa_list or []
        self.curvature = curvature
        self._taxa_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(self.taxa_list)
        }

        # Standard nn.Embedding; values will be projected onto Poincaré ball
        # after loading pretrained weights
        self.embeddings = nn.Embedding(n_taxa, dim)
        nn.init.uniform_(self.embeddings.weight, -0.001, 0.001)

    def taxon_to_id(self, taxon_name: str) -> int:
        """Convert a taxon name string to an integer ID (0-indexed)."""
        idx = self._taxa_to_idx.get(taxon_name, -1)
        if idx == -1:
            # Fall back to index 0 (unknown taxon)
            return 0
        return idx

    @classmethod
    def from_distance_matrix(
        cls,
        dist_matrix,
        taxa_list: List[str],
        dim: int = 64,
        epochs: int = 200,
    ) -> "PhyloEmbedding":
        """
        Train Poincaré embeddings from a pairwise distance matrix.

        Parameters
        ----------
        dist_matrix : np.ndarray[N, N]
        taxa_list   : list[str]         N taxon names
        dim         : int
        epochs      : int

        Returns
        -------
        PhyloEmbedding (with trained weights)
        """
        from src.utils.phylo import PoincareEmbeddingTrainer
        trainer = PoincareEmbeddingTrainer(dim=dim, epochs=epochs)
        return trainer.train(dist_matrix, taxa_list, verbose=False)

    def forward(self, taxon_ids: Tensor) -> Tensor:
        """
        Parameters
        ----------
        taxon_ids : Tensor[B] integer taxon indices

        Returns
        -------
        Tensor[B, dim]
        """
        emb = self.embeddings(taxon_ids)  # [B, dim]
        # Project onto Poincaré ball via scaled tanh (ensures ||emb||_2 < 1/sqrt(c))
        try:
            import geoopt
            manifold = geoopt.PoincareBall(c=self.curvature)
            # expmap from origin to ensure we stay on the manifold
            emb = manifold.expmap0(emb)
        except ImportError:
            # If geoopt unavailable, use plain embedding with l2 normalisation guard
            norm = emb.norm(dim=-1, keepdim=True).clamp(min=1.0)
            emb = emb / norm * 0.99
        return emb


# ---------------------------------------------------------------------------
# MorphoBERT
# ---------------------------------------------------------------------------

class MorphoBERT(nn.Module):
    """
    BioBERT-based morphological text encoder.

    Wraps dmis-lab/biobert-base-cased-v1.2 with a linear projection head.

    Parameters
    ----------
    out_dim       : int    output projection dimension
    pretrained_id : str    HuggingFace model identifier
    freeze_base   : bool   if True, freeze all BioBERT weights (only train projection)
    """

    def __init__(
        self,
        out_dim: int = 256,
        pretrained_id: str = "dmis-lab/biobert-base-cased-v1.2",
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self._loaded = False

        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_id)
            self.bert = AutoModel.from_pretrained(pretrained_id)
            if freeze_base:
                for param in self.bert.parameters():
                    param.requires_grad = False
            self._loaded = True
            logger.info("Loaded BioBERT from %s", pretrained_id)
        except Exception as exc:
            logger.warning(
                "Could not load BioBERT (%s). MorphoBERT will return zeros. "
                "Install transformers and download the model to enable it.",
                exc,
            )
            self.bert = None
            self.tokenizer = None

        # Projection head: 768 → out_dim
        self.proj = nn.Sequential(
            nn.Linear(768, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def encode_text(self, texts: List[str], device) -> Tensor:
        """Tokenise and encode a list of text strings."""
        if not self._loaded or self.tokenizer is None:
            B = len(texts)
            return torch.zeros(B, 768, device=device)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.set_grad_enabled(self.training):
            out = self.bert(**enc)
        # Use [CLS] token representation
        return out.last_hidden_state[:, 0, :]  # [B, 768]

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input_ids      : Tensor[B, seq_len]
        attention_mask : Tensor[B, seq_len]

        Returns
        -------
        Tensor[B, out_dim]
        """
        if not self._loaded or self.bert is None:
            B = input_ids.shape[0]
            return torch.zeros(B, self.out_dim, device=input_ids.device)

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = out.last_hidden_state[:, 0, :]   # [B, 768]
        return self.proj(cls_feat)


# ---------------------------------------------------------------------------
# CLIPImageEncoder
# ---------------------------------------------------------------------------

class CLIPImageEncoder(nn.Module):
    """
    CLIP vision encoder for optional reference bone images.

    When no image is provided, returns zeros (handled gracefully by
    TaxonomyEncoder's null-conditioning mechanism).

    Parameters
    ----------
    out_dim       : int    output projection dimension
    pretrained_id : str    HuggingFace CLIP model identifier
    freeze_base   : bool   freeze CLIP weights
    """

    def __init__(
        self,
        out_dim: int = 256,
        pretrained_id: str = "openai/clip-vit-base-patch32",
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self._loaded = False

        clip_hidden = 768  # ViT-B/32 default
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(pretrained_id)
            # Load the full CLIPModel and extract just the vision encoder
            full_clip = CLIPModel.from_pretrained(pretrained_id)
            self.clip = full_clip.vision_model
            clip_hidden = full_clip.config.vision_config.hidden_size
            del full_clip
            if freeze_base:
                for param in self.clip.parameters():
                    param.requires_grad = False
            self._loaded = True
            logger.info("Loaded CLIP vision encoder from %s (hidden=%d)", pretrained_id, clip_hidden)
        except Exception as exc:
            logger.warning(
                "Could not load CLIP (%s). CLIPImageEncoder will return zeros.",
                exc,
            )
            self.clip = None
            self.processor = None

        # Projection: clip_hidden → out_dim
        self.proj = nn.Sequential(
            nn.Linear(clip_hidden, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, pixel_values: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        pixel_values : Tensor[B, 3, H, W] or None

        Returns
        -------
        Tensor[B, out_dim]   (zeros if pixel_values is None or CLIP unavailable)
        """
        if pixel_values is None or not self._loaded or self.clip is None:
            # Return zeros for null conditioning
            B = 1 if pixel_values is None else pixel_values.shape[0]
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cpu")
            return torch.zeros(B, self.out_dim, device=device)

        B = pixel_values.shape[0]
        with torch.set_grad_enabled(self.training):
            out = self.clip(pixel_values=pixel_values)
        # Use pooled output
        pooled = out.pooler_output  # [B, 768]
        return self.proj(pooled)    # [B, out_dim]


# ---------------------------------------------------------------------------
# TaxonomyEncoder
# ---------------------------------------------------------------------------

class TaxonomyEncoder(nn.Module):
    """
    Combined conditioning encoder.

    Concatenates:
        PhyloEmbedding(64) + MorphoBERT(256) + CLIPImageEncoder(256) = 576
    → Linear → 256

    Handles null inputs for any modality gracefully (replaces missing inputs
    with learned null embeddings for classifier-free guidance).

    Parameters
    ----------
    n_taxa    : int    number of taxa in the embedding table
    out_dim   : int    final conditioning dimension (default 256)
    taxa_list : list   ordered taxon name list
    """

    def __init__(
        self,
        n_taxa: int = 1024,
        out_dim: int = 256,
        taxa_list: Optional[List[str]] = None,
        phylo_dim: int = 64,
        morpho_dim: int = 256,
        image_dim: int = 256,
        freeze_bert: bool = False,
        freeze_clip: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.phylo_dim = phylo_dim
        self.morpho_dim = morpho_dim
        self.image_dim = image_dim

        self.phylo_enc = PhyloEmbedding(
            n_taxa=n_taxa, dim=phylo_dim, taxa_list=taxa_list or []
        )
        self.morpho_enc = MorphoBERT(out_dim=morpho_dim, freeze_base=freeze_bert)
        self.image_enc = CLIPImageEncoder(out_dim=image_dim, freeze_base=freeze_clip)

        in_dim = phylo_dim + morpho_dim + image_dim
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

        # Learnable null embeddings for classifier-free guidance
        self.null_phylo = nn.Parameter(torch.zeros(phylo_dim))
        self.null_morpho = nn.Parameter(torch.zeros(morpho_dim))
        self.null_image = nn.Parameter(torch.zeros(image_dim))

    def forward(
        self,
        taxon_ids: Optional[Tensor] = None,              # [B] int
        input_ids: Optional[Tensor] = None,              # [B, seq_len]
        attention_mask: Optional[Tensor] = None,         # [B, seq_len]
        pixel_values: Optional[Tensor] = None,           # [B, 3, H, W]
        use_null_conditioning: bool = False,
    ) -> Tensor:
        """
        Parameters
        ----------
        taxon_ids            : int tensor [B] or None → phylo embedding
        input_ids            : BERT tokens [B, L] or None
        attention_mask       : BERT mask [B, L] or None
        pixel_values         : image pixels [B, 3, H, W] or None
        use_null_conditioning: if True, returns all-null embedding (CFG)

        Returns
        -------
        Tensor[B, out_dim]
        """
        # Determine batch size from any available input
        B = 1
        device = self.null_phylo.device
        if taxon_ids is not None:
            B = taxon_ids.shape[0]
            device = taxon_ids.device
        elif input_ids is not None:
            B = input_ids.shape[0]
            device = input_ids.device
        elif pixel_values is not None:
            B = pixel_values.shape[0]
            device = pixel_values.device

        if use_null_conditioning:
            phylo_feat = self.null_phylo.unsqueeze(0).expand(B, -1)
            morpho_feat = self.null_morpho.unsqueeze(0).expand(B, -1)
            image_feat = self.null_image.unsqueeze(0).expand(B, -1)
        else:
            # Phylo
            if taxon_ids is not None:
                phylo_feat = self.phylo_enc(taxon_ids.to(device))
            else:
                phylo_feat = self.null_phylo.unsqueeze(0).expand(B, -1)

            # Morpho (BioBERT)
            if input_ids is not None and attention_mask is not None:
                morpho_feat = self.morpho_enc(
                    input_ids.to(device), attention_mask.to(device)
                )
            else:
                morpho_feat = self.null_morpho.unsqueeze(0).expand(B, -1)

            # Image (CLIP)
            if pixel_values is not None:
                image_feat = self.image_enc(pixel_values.to(device))
            else:
                image_feat = self.null_image.unsqueeze(0).expand(B, -1)

        # Concatenate and project
        combined = torch.cat([phylo_feat, morpho_feat, image_feat], dim=-1)  # [B, in_dim]
        return self.fusion(combined)
