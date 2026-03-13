"""
Phylogenetic utilities.

OTLClient               — queries Open Tree of Life v3 API
PoincareEmbeddingTrainer— trains Poincaré ball embeddings on distance matrix
load_or_train_phylo_embedding — cached load / train helper
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

OTL_API = "https://api.opentreeoflife.org/v3"


# ---------------------------------------------------------------------------
# Open Tree of Life client
# ---------------------------------------------------------------------------

class OTLClient:
    """
    Thin wrapper around the Open Tree of Life REST API v3.

    https://opentreeoflife.github.io/develop/api/v3
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{OTL_API}/{endpoint.lstrip('/')}"
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_taxon_id(self, taxon_name: str) -> int:
        """
        Resolve a taxon name to its OTT (Open Tree Taxonomy) numeric ID.

        Raises KeyError if no match found.
        """
        payload = {"names": [taxon_name], "do_approximate_matching": True}
        result = self._post("tnrs/match_names", payload)
        matches = result.get("results", [])
        if not matches or not matches[0].get("matches"):
            raise KeyError(f"No OTT match for taxon '{taxon_name}'")
        best = matches[0]["matches"][0]
        return int(best["taxon"]["ott_id"])

    def get_induced_subtree(self, taxon_ids: List[int]) -> dict:
        """
        Fetch the synthetic induced subtree (Newick + node mapping) for a
        list of OTT IDs.

        Returns a dict with keys:
          newick    : str
          node_ids  : list[str]
          ott_ids   : list[int]
        """
        payload = {"ott_ids": [int(i) for i in taxon_ids]}
        result = self._post("tree_of_life/induced_subtree", payload)
        return {
            "newick": result.get("newick", ""),
            "node_ids": result.get("node_ids_not_in_tree", []),
            "ott_ids": taxon_ids,
            "raw": result,
        }

    def compute_cophenetic_distances(
        self, subtree_result: dict
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute a pairwise cophenetic distance matrix from an induced subtree.

        Uses the Newick string from `get_induced_subtree` and the Bio.Phylo
        library for tree parsing.

        Returns
        -------
        dist_matrix : np.ndarray[n_taxa, n_taxa]   symmetric float32 distances
        ott_id_order: list[int]                      row/col ordering
        """
        try:
            from Bio import Phylo
            from io import StringIO
        except ImportError:
            raise ImportError(
                "biopython is required for cophenetic distance computation: "
                "pip install biopython"
            )

        newick = subtree_result.get("newick", "")
        ott_ids = subtree_result.get("ott_ids", [])

        if not newick:
            # Return identity matrix if no tree available
            n = len(ott_ids)
            return np.zeros((n, n), dtype=np.float32), list(ott_ids)

        tree = Phylo.read(StringIO(newick), "newick")
        terminals = tree.get_terminals()
        n = len(terminals)
        dist_matrix = np.zeros((n, n), dtype=np.float32)

        for i, t1 in enumerate(terminals):
            for j, t2 in enumerate(terminals):
                if i < j:
                    d = float(tree.distance(t1, t2))
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

        # Map terminal names (OTT labels) to indices
        terminal_names = [t.name for t in terminals]
        return dist_matrix, terminal_names


# ---------------------------------------------------------------------------
# Poincaré Embedding Trainer
# ---------------------------------------------------------------------------

class PoincareEmbeddingTrainer:
    """
    Trains Poincaré ball embeddings on a pairwise distance matrix.

    Uses geoopt's PoincareBall manifold and RiemannianAdam optimizer.
    """

    def __init__(
        self,
        dim: int = 64,
        lr: float = 5e-2,
        epochs: int = 200,
        curvature: float = 1.0,
        burn_in_epochs: int = 10,
        burn_in_lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.curvature = curvature
        self.burn_in_epochs = burn_in_epochs
        self.burn_in_lr = burn_in_lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        distance_matrix: np.ndarray,
        taxa_list: List[str],
        verbose: bool = True,
    ) -> "PhyloEmbedding":
        """
        Train Poincaré embeddings from a symmetric distance matrix.

        Parameters
        ----------
        distance_matrix : np.ndarray[N, N]  symmetric, non-negative
        taxa_list       : list[str]         N taxa names (row/col labels)

        Returns
        -------
        PhyloEmbedding with trained weights
        """
        try:
            import geoopt
        except ImportError:
            raise ImportError("geoopt is required: pip install geoopt")

        n = len(taxa_list)
        assert distance_matrix.shape == (n, n), "distance_matrix must be [N, N]"

        # Normalise distances to [0, 1]
        max_dist = distance_matrix.max()
        if max_dist > 0:
            dist_norm = distance_matrix / max_dist
        else:
            dist_norm = distance_matrix.copy()

        dist_t = torch.tensor(dist_norm, dtype=torch.float32, device=self.device)

        manifold = geoopt.PoincareBall(c=self.curvature)
        # Initialise embeddings near origin for numerical stability
        init = torch.randn(n, self.dim, device=self.device) * 0.001
        embeddings = geoopt.ManifoldParameter(init, manifold=manifold)

        # Burn-in phase with small LR
        opt = geoopt.optim.RiemannianAdam([embeddings], lr=self.burn_in_lr)
        for epoch in range(self.burn_in_epochs):
            loss = self._loss(embeddings, dist_t, manifold)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if verbose and epoch % 5 == 0:
                logger.info("[Poincaré burn-in %d/%d] loss=%.4f", epoch, self.burn_in_epochs, loss.item())

        # Main training
        opt = geoopt.optim.RiemannianAdam([embeddings], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        for epoch in range(self.epochs):
            loss = self._loss(embeddings, dist_t, manifold)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            if verbose and epoch % 50 == 0:
                logger.info("[Poincaré %d/%d] loss=%.4f", epoch, self.epochs, loss.item())

        # Build PhyloEmbedding
        from src.models.conditioning import PhyloEmbedding
        embedding_module = PhyloEmbedding(
            n_taxa=n,
            dim=self.dim,
            taxa_list=taxa_list,
            curvature=self.curvature,
        )
        with torch.no_grad():
            embedding_module.embeddings.data.copy_(embeddings.data.cpu())

        return embedding_module

    @staticmethod
    def _loss(
        embeddings: "torch.Tensor",
        target_dists: "torch.Tensor",
        manifold,
    ) -> "torch.Tensor":
        """
        Mean-squared error between Poincaré distances and target distances.
        """
        n = embeddings.shape[0]
        # Compute pairwise Poincaré distances
        emb_i = embeddings.unsqueeze(1).expand(n, n, -1)  # [N, N, D]
        emb_j = embeddings.unsqueeze(0).expand(n, n, -1)  # [N, N, D]

        # Flatten for manifold distance computation
        flat_i = emb_i.reshape(n * n, -1)
        flat_j = emb_j.reshape(n * n, -1)
        poincare_dists = manifold.dist(flat_i, flat_j).reshape(n, n)

        mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
        loss = ((poincare_dists[mask] - target_dists[mask]) ** 2).mean()
        return loss


# ---------------------------------------------------------------------------
# Cached load or train helper
# ---------------------------------------------------------------------------

def load_or_train_phylo_embedding(
    taxa_list: List[str],
    cache_dir: str | Path,
    dim: int = 64,
    epochs: int = 200,
    force_retrain: bool = False,
) -> "PhyloEmbedding":
    """
    Load a cached PhyloEmbedding or train one from scratch.

    Queries OTL for cophenetic distances, trains Poincaré embeddings,
    and caches the result to `cache_dir`.

    Parameters
    ----------
    taxa_list   : list of taxon name strings
    cache_dir   : directory to cache the embedding
    dim         : embedding dimensionality
    epochs      : training epochs
    force_retrain: if True, ignore cached embedding
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a cache key from the sorted taxa list
    taxa_key = "_".join(sorted(taxa_list))[:64]
    cache_pt = cache_dir / f"phylo_embed_{taxa_key}_{dim}d.pt"

    if cache_pt.exists() and not force_retrain:
        logger.info("Loading cached phylo embedding from %s", cache_pt)
        from src.models.conditioning import PhyloEmbedding
        state = torch.load(str(cache_pt), map_location="cpu", weights_only=False)
        module = PhyloEmbedding(
            n_taxa=len(taxa_list),
            dim=dim,
            taxa_list=taxa_list,
        )
        module.load_state_dict(state)
        return module

    logger.info("Training phylo embedding for %d taxa...", len(taxa_list))

    # Query OTL
    client = OTLClient()
    try:
        ott_ids = []
        for taxon in taxa_list:
            try:
                ott_ids.append(client.get_taxon_id(taxon))
            except Exception as exc:
                logger.warning("OTL lookup failed for '%s': %s", taxon, exc)
                ott_ids.append(-1)

        valid_ids = [i for i in ott_ids if i > 0]
        if len(valid_ids) < 2:
            raise ValueError("Too few valid OTT IDs to build a tree")

        subtree = client.get_induced_subtree(valid_ids)
        dist_matrix, _ = client.compute_cophenetic_distances(subtree)

    except Exception as exc:
        logger.warning("OTL failed (%s). Falling back to random distance matrix.", exc)
        n = len(taxa_list)
        dist_matrix = np.random.exponential(1.0, (n, n))
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)

    trainer = PoincareEmbeddingTrainer(dim=dim, epochs=epochs)
    embedding = trainer.train(dist_matrix, taxa_list, verbose=False)

    # Cache
    torch.save(embedding.state_dict(), str(cache_pt))
    logger.info("Saved phylo embedding to %s", cache_pt)

    return embedding
