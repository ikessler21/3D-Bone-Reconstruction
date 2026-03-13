---
name: 3d-fossil-reconstruction-plan
overview: |
  A PyTorch-based framework for 3D fossil bone reconstruction from fragmentary specimens.
  Novel contributions: (1) RELIC (Reconstruction of Extinct Life via Informed Completion) —
  a phylogenetically-conditioned latent diffusion model with confidence-gated bilateral
  symmetry constraints; (2) PaleoComplete — the first dedicated benchmark for paleontological
  shape completion using landmark-based completeness levels; (3) a three-phase domain
  adaptation pipeline tailored to fossil scanning artifacts. Framed as a domain inductive
  bias paper: we formalize the morphological priors specific to fossil bone completion and
  show their consistent benefit over strong general-purpose backbones.
todos:
  - id: data-survey
    content: |
      Acquire and audit open-access 3D fossil/bone datasets using a two-tier quality system.
      Apply keyword filter to all specimens (exclude "reconstruction", "cast", "restored",
      "plaster", "composite", "missing", "juvenile", "subadult", "hatchling", "fetal" via
      pyMorphoSource API). Assign completeness_confidence score (0–1) to all specimens.
      Tier A (test split only): keyword filter + curvature anomaly detection (flag >5%
      near-zero Gaussian curvature patches) + literature cross-reference; confidence >= 0.8.
      Tier B (training/validation): keyword filter only; confidence >= 0.5. Priority sources:
      MorphoSource (~64K open-DL), Phenome10K (1,640+ STLs), Dryad CC0 fossils, Smithsonian
      3D (CC0), and proxy datasets: VerSe vertebrae, ICL femur/tibia meshes, BoneDat pelvis.
    status: pending
  - id: phylo-embedding
    content: |
      Build the phylogenetic + morphological conditioning encoder (replaces CLIP text).
      (1) Phylogenetic position: query Open Tree of Life API via python-opentree to get
      induced subtrees for all training taxa; compute pairwise cophenetic distances; train
      a Poincaré ball embedding (geoopt, dim=64) on the distance matrix. Each taxon gets
      a fixed hyperbolic vector. (2) Morphological descriptor: fine-tune BioBERT on
      MorphoSource project descriptions + paleontological literature abstracts to understand
      clade names, bone type, anatomical side, stratigraphic terms. Keep CLIP image encoder
      only for the optional reference image branch.
    status: pending
  - id: paleocomplete-benchmark
    content: |
      Construct PaleoComplete benchmark with landmark-based completeness levels (not % missing).
      Define anatomical landmark sets per bone type (e.g., femur: proximal epiphysis, lesser
      trochanter, diaphysis midpoint, distal medial condyle, distal lateral condyle). Score
      completeness as fraction of landmark regions present: L1 (1–2 landmarks missing),
      L2 (3 landmarks missing), L3 (4+ landmarks missing). Curate ~500+ high-confidence-
      complete specimens. Generate standardized partial shards per level with fixed seeds.
      Store as {partial_points, full_points, metadata} including ontogenetic_stage,
      completeness_confidence, taxon, bone_type, geological_age, museum_catalog_number.
      Hold out test split; release with Papers With Code leaderboard.
    status: pending
  - id: data-pipeline
    content: |
      Implement preprocessing pipeline: mesh-to-point-cloud (Poisson disk + area-weighted
      sampling, 4096 pts), PCA normalization, ontogenetic stage filtering (adult-only default),
      taphonomic deformation augmentation (TPS compression, affine shear, surface erosion via
      SinPoint-style smooth displacement fields), CT artifact augmentation (ring noise, matrix
      contamination, mineralization infilling, resolution dropout), landmark-based partial shard
      generation. Exclude specimens with PCA aspect-ratio deviation >20% from taxon mean
      (flagged as taphonomically deformed beyond scope). Store as .pt files with metadata.
    status: pending
  - id: model-baseline
    content: |
      Implement AdaPoinTr-based deterministic baseline adapted for bone morphology: add
      confidence-gated symmetry loss, normal consistency loss, phylogenetic class conditioning
      (discrete bone-type + order embedding). Evaluate on PaleoComplete and PCN. Compare
      against PointNet AE, PoinTr, SeedFormer, DiffComplete.
    status: pending
  - id: model-relic
    content: |
      Implement RELIC. Architecture: LION-style two-stage latent diffusion. Stage 1: hierarchical
      VAE (AdaPoinTr encoder → z_global ∈ ℝ^256 + z_local ∈ ℝ^(K×d) → SeedFormer decoder;
      trained with Chamfer L1 + KL). Stage 2: DDPM in z_global latent space conditioned via
      cross-attention on (a) partial cloud encoder features, (b) Poincaré phylogenetic embedding,
      (c) BioBERT morphological embedding, (d) optional CLIP reference image. DDIM 20-step
      fast sampling at inference. Confidence-gated symmetry module injected into encoder.
      Null-conditioning for classifier-free guidance at inference.
    status: pending
  - id: symmetry-module
    content: |
      Implement confidence-gated bilateral symmetry module. RANSAC over candidate symmetry
      planes → returns (plane_normal, plane_offset, confidence ∈ [0,1]) where confidence =
      fraction of points with a mirror-match within threshold. Symmetry loss = confidence *
      CD(pred, reflect(pred, plane)). When confidence < 0.25 the loss drops to zero (graceful
      fallback for asymmetric/severely fragmentary specimens). Report BSE separately for
      high-confidence symmetric bones vs. low-confidence/asymmetric ones as an ablation result.
    status: pending
  - id: domain-adaptation
    content: |
      Implement sequential three-phase domain adaptation pipeline.
      Phase 1 (during main training, free): CT artifact augmentations from data-pipeline.
      Phase 2 (adversarial, 10–15 epochs, unpaired real scans): freeze decoder; fine-tune
      encoder with GRL domain classifier (real vs synthetic); data = 500–1000 unpaired
      MorphoSource CT-derived meshes.
      Phase 3 (self-supervised, 5–10 epochs, unpaired real scans): masked autoencoding on
      real scans (mask 40% → reconstruct); consistency loss between two random maskings of
      same specimen when multiple scans available; no ground-truth required.
      Phases run sequentially, not simultaneously.
    status: pending
  - id: training-eval
    content: |
      Multi-metric evaluation suite: CD-L1, CD-L2, F-Score@1%, Normal Consistency, BSE
      (reported separately for symmetric vs asymmetric bones), diversity (MMD, COV across
      k=10 samples). Add uncertainty calibration: Spearman correlation between per-point std
      across diffusion samples and per-point reconstruction error — report ρ to validate that
      uncertainty is meaningful, not decorative. Standard benchmark: run on PCN dataset (8
      categories) to contextualize model quality for CV reviewers. Full ablation table including
      pre-training strategy row (no pre-train / PCN pre-train / bone pre-train).
    status: pending
  - id: expert-loop
    content: |
      Prototype expert-in-the-loop Gradio interface with calibrated uncertainty heatmap
      (colored by per-point std, validated via Spearman correlation). User study: recruit
      minimum 15 domain-matched vertebrate paleontologists (specialists matched to bone/taxa
      type); use 2AFC forced-choice protocol (which of A/B is more plausible?) on 20 test
      specimens; pre-register protocol on OSF before running; report inter-rater reliability
      (Krippendorff's α). Recruit via Society of Vertebrate Paleontology mailing list.
    status: pending
  - id: paper-writeup
    content: |
      Prepare research paper with explicit framing: RELIC does not propose a novel general
      architecture — it identifies and formalizes the domain-specific inductive biases
      required for paleontological bone completion (phylogenetic shape priors, confidence-
      gated bilateral symmetry, fossil scan domain adaptation) and demonstrates their
      consistent benefit over strong general backbones on a purpose-built benchmark.
      Target venues: Methods in Ecology and Evolution / PLOS Computational Biology
      (paleontology-primary) or CVPR/ICCV workshop track (CV-primary). Release code,
      PaleoComplete benchmark, pretrained weights, and Gradio demo.
    status: pending
isProject: true
---

## 1. Project scope and research contributions

### Goal

Given a partial 3D point cloud of a fragmentary fossil bone (specifically specimens with
low taphonomic geometric deformation — fragmentary incompleteness, not crushing/shearing),
produce one or more plausible complete reconstructions while (a) respecting bilateral
morphological symmetry where applicable and with calibrated confidence, (b) conditioning
on phylogenetic position and morphological descriptors, and (c) quantifying reconstruction
uncertainty with validated calibration for downstream expert review.

### Explicit scope limitation

RELIC addresses **fragmentary incompleteness** — missing spatial regions — in specimens
with ≤20% geometric distortion from their taxon mean shape (estimated via PCA aspect-ratio
heuristic). Specimens with severe taphonomic deformation (compression, shearing) are out
of scope and excluded from training and evaluation. This is a known, documented limitation.

### Novel contributions (paper-ready)


| #   | Contribution                                                                                             | Significance                                                                                |
| --- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 1   | **RELIC** — phylogenetically-conditioned latent diffusion model with confidence-gated symmetry           | First dedicated generative model for paleontological 3D reconstruction                      |
| 2   | **PaleoComplete benchmark** — landmark-based completeness levels, curated ground truth, leaderboard      | First benchmark; enables future comparative evaluation                                      |
| 3   | **Confidence-gated bilateral symmetry module** — RANSAC plane + equivariant encoder + soft mirror loss   | Gracefully handles asymmetric/deformed specimens; novel BSE metric                          |
| 4   | **Poincaré phylogenetic + BioBERT morphological conditioning**                                           | Replaces naive text encoding; first use of hyperbolic phylo embeddings for shape completion |
| 5   | **Three-phase fossil domain adaptation** — CT augmentation + adversarial alignment + masked autoencoding | Concrete sequential algorithm; demonstrated on real MorphoSource scans                      |


### Framing (for paper positioning)

> *RELIC does not propose a novel general-purpose architecture. Instead, we identify and
> formalize the domain-specific inductive biases required for paleontological bone completion
> — phylogenetic shape priors, conditional bilateral symmetry, and fossil scan artifacts —
> and show that incorporating these biases into a strong general backbone (AdaPoinTr +
> latent diffusion) yields significant, consistent improvements on PaleoComplete and
> competitive performance on the standard PCN benchmark.*

---

## 2. Repository structure

```
bones/
├── data/
│   ├── raw/                          # downloaded meshes (MorphoSource, Phenome10K, etc.)
│   ├── processed/                    # normalized point clouds as .pt files
│   └── paleocomplete/                # PaleoComplete benchmark splits
├── src/
│   ├── datasets/
│   │   ├── morphosource.py           # pyMorphoSource download + curation filter
│   │   ├── fossil_dataset.py         # PyTorch Dataset: {partial, full, metadata}
│   │   └── augmentations.py          # landmark shards, CT noise, TPS deformation
│   ├── models/
│   │   ├── encoder.py                # AdaPoinTr-style geometry-aware encoder
│   │   ├── symmetry.py               # confidence-gated RANSAC symmetry module
│   │   ├── vae.py                    # hierarchical VAE (Stage 1)
│   │   ├── diffusion.py              # DDPM/DDIM in z_global latent space (Stage 2)
│   │   ├── conditioning.py           # Poincaré phylo embedding + BioBERT morpho encoder
│   │   ├── decoder.py                # SeedFormer-style hierarchical transformer decoder
│   │   └── relic.py                  # full RELIC model (Stage 1 + Stage 2)
│   ├── training/
│   │   ├── losses.py                 # CD, F-Score, normal consistency, gated symmetry loss
│   │   ├── trainer.py                # training loop, checkpointing, mixed precision
│   │   └── configs/                  # YAML experiment configs
│   ├── eval/
│   │   ├── metrics.py                # CD, F-Score, NC, BSE, diversity, calibration (Spearman ρ)
│   │   ├── evaluate.py               # benchmark evaluation (PaleoComplete + PCN)
│   │   └── visualize.py              # Open3D renders, uncertainty heatmaps, t-SNE
│   ├── domain_adaptation/
│   │   ├── ct_augmentations.py       # ring artifacts, matrix noise, mineralization, erosion
│   │   └── adaptation.py             # Phase 2 GRL adversarial + Phase 3 masked autoencoding
│   └── utils/
│       ├── geometry.py               # sampling, normalization, PCA alignment, landmark scoring
│       ├── phylo.py                  # OTL API queries, cophenetic distance matrix, geoopt embed
│       └── io.py                     # PLY, OBJ, STL, NIfTI, VTK I/O
├── experiments/                      # YAML configs per ablation/run
├── notebooks/                        # exploratory analysis and visualization
├── expert_interface/                 # Gradio expert-in-the-loop UI
├── paleocomplete/                    # benchmark release artifacts + eval script
├── pyproject.toml
└── README.md
```

---

## 3. Datasets

### 3.1 Primary fossil datasets


| Dataset                               | URL                                                                                | # Specimens                             | Formats              | License           | Priority      |
| ------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------- | -------------------- | ----------------- | ------------- |
| **MorphoSource**                      | [https://www.morphosource.org](https://www.morphosource.org)                       | ~64,552 open-DL; ~1,224 fossil-tagged   | OBJ, PLY, STL, DICOM | CC-BY-NC (varies) | **High**      |
| **Phenome10K**                        | [https://www.phenome10k.org](https://www.phenome10k.org)                           | 1,640+                                  | STL                  | CC BY-NC 4.0      | **High**      |
| **Smithsonian 3D**                    | [https://3d.si.edu](https://3d.si.edu)                                             | Dozens (dinosaurs, cetaceans, hominins) | OBJ, STL             | CC0               | High          |
| **Dryad: Devonian fish skull**        | [https://doi.org/10.5061/dryad.n66h4](https://doi.org/10.5061/dryad.n66h4)         | 4 PLY meshes                            | PLY                  | **CC0**           | High          |
| **Dryad: Paleocene Glires mandibles** | [https://doi.org/10.5061/dryad.69p8cz95g](https://doi.org/10.5061/dryad.69p8cz95g) | 5 specimens                             | STL, PLY             | **CC0**           | High          |
| **Zenodo: Equus stenonis skull**      | [https://doi.org/10.5281/zenodo.3895217](https://doi.org/10.5281/zenodo.3895217)   | 6 PLY meshes                            | PLY                  | CC-BY-4.0         | Medium        |
| **Zenodo: Isthminia dolphin**         | [https://doi.org/10.5281/zenodo.27214](https://doi.org/10.5281/zenodo.27214)       | 6 models                                | DICOM, STL, OBJ      | CC-BY-NC-4.0      | Medium        |
| **Zenodo: Enantiornithine teeth**     | [https://doi.org/10.5281/zenodo.5502305](https://doi.org/10.5281/zenodo.5502305)   | 3 specimens                             | DICOM, STL           | CC-BY-4.0         | Medium        |
| **DigiMorph**                         | [http://digimorph.org](http://digimorph.org)                                       | 1,000+                                  | STL                  | Non-commercial    | Low (license) |


**MorphoSource batch download:** [https://github.com/JulieWinchester/morphosource-scrape](https://github.com/JulieWinchester/morphosource-scrape)
**Fossil mesh search (open access):** `https://www.morphosource.org/catalog/media?q=fossil&open_access=1&media_type=Mesh`

### 3.2 Proxy bone datasets (pre-training / domain transfer)


| Dataset             | URL                                                                                | # Specimens         | Formats      | License      | Bone Type             |
| ------------------- | ---------------------------------------------------------------------------------- | ------------------- | ------------ | ------------ | --------------------- |
| **VerSe vertebrae** | [https://github.com/anjany/verse](https://github.com/anjany/verse)                 | 374 CT scans        | NIfTI, JSON  | CC-BY-SA-4.0 | C1–L6 vertebrae       |
| **RibSeg v2**       | [https://github.com/HINTLab/RibSeg](https://github.com/HINTLab/RibSeg)             | 490 CT, 11,719 ribs | NIfTI, NumPy | Apache-2.0   | Ribs                  |
| **BoneDat**         | [https://doi.org/10.5281/zenodo.15189761](https://doi.org/10.5281/zenodo.15189761) | 278 CT scans        | NIfTI, VTK   | CC-BY-4.0    | Pelvis, sacrum, L4/L5 |
| **ICL Femur/Tibia** | [https://doi.org/10.5281/zenodo.167808](https://doi.org/10.5281/zenodo.167808)     | 70 STL meshes       | STL          | CC-BY-4.0    | Femur, tibia/fibula   |


**VerSe direct downloads:**

```
https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip
https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip
```

**ICL Femur/Tibia direct downloads:**

```
https://zenodo.org/record/167808/files/femur_3D_surface_meshes.zip
https://zenodo.org/record/167808/files/tibia_3D_surface_meshes.zip
```

### 3.3 Ground-truth quality audit — two-tier system

Many museum specimens labeled "complete" are partially restored with plaster or 3D-printed
infills, often undocumented. Quality requirements differ by use: training data can tolerate
noise; the benchmark test split cannot. Specimens are assigned to one of two tiers.

**Tier A — strict (test split only):**

1. **Metadata keyword filter** (via `pyMorphoSource` API): exclude any specimen or project
  description containing `"reconstruction"`, `"cast"`, `"restored"`, `"plaster"`,
   `"composite"`, `"missing"`, `"juvenile"`, `"subadult"`, `"hatchling"`, `"fetal"`.
2. **Curvature anomaly detection**: compute per-point Gaussian curvature on the mesh;
  flag specimens where >5% of surface area has near-zero curvature in contiguous patches
  > 1cm² (plaster fills are anomalously smooth relative to true bone texture).
3. **Literature cross-reference**: manually verify each specimen against its describing
  publication.
4. **completeness_confidence ≥ 0.8** required to enter the PaleoComplete test split.

Tier A will yield a small but honest test set (~50–100 specimens initially). This is
intentional. A small, verified test set is preferable to a large, questionable one.

**Tier B — relaxed (training and validation splits):**

1. **Metadata keyword filter only** (same keyword list as Tier A) — automated, no manual
  verification required.
2. **completeness_confidence ≥ 0.5** (computed from keyword filter + curvature score alone).
3. No literature cross-reference required.

Tier B admits noisier specimens (possible undocumented fills, minor composites). This is
acceptable for training: noisy ground truth adds variance but does not bias the model
systematically, and is partially mitigated by augmentation. When museum partnerships provide
cleaner data, the training split can be upgraded and the model retrained.

**Output**: each specimen receives a `completeness_confidence` score (0–1) and a tier
assignment (A or B) stored in metadata. All benchmark results report test-split tier
explicitly. Paper framing: *"PaleoComplete v0.1 is a curated pilot benchmark; community
contributions and museum-partnered expansions are planned."*

### 3.4 Data curation strategy

1. **Phase 1 (bootstrap):** ICL femur/tibia (70) + VerSe vertebrae (374) + BoneDat
  pelvis (278) — clean, adult, CC-licensed → initial pre-training. Ablate this against
   random init and PCN pre-train to verify positive transfer.
2. **Phase 2 (paleontology-specific):** Tier B MorphoSource + Phenome10K + Dryad CC0
  fossils → fine-tune; Tier A subset → PaleoComplete test split.
3. **Phase 3 (real-scan adaptation):** Unpaired MorphoSource CT-derived meshes (500–1000)
  → domain adaptation training (no ground-truth required).

Splits: stratified by taxonomic order AND bone type AND ontogenetic stage; no specimen
overlap across splits. Tier A specimens appear only in the test split.

---

## 4. PaleoComplete benchmark (novel dataset contribution)

### Design — landmark-based completeness

Rather than arbitrary % missing, completeness is defined by which anatomical landmarks
are present — matching how paleontologists actually describe preservation.

**Example landmark sets:**


| Bone type | Landmarks (5 per bone)                                                                                   |
| --------- | -------------------------------------------------------------------------------------------------------- |
| Femur     | Proximal epiphysis, lesser trochanter, diaphysis midpoint, distal medial condyle, distal lateral condyle |
| Tibia     | Proximal plateau, tibial tuberosity, diaphysis midpoint, distal medial malleolus, distal fibular notch   |
| Vertebra  | Neural spine, left transverse process, right transverse process, centrum anterior, centrum posterior     |


**Completeness levels:**

- **L1**: 1–2 landmarks missing (mild fragmentation)
- **L2**: 3 landmarks missing (moderate)
- **L3**: 4+ landmarks missing (severe — scientifically most challenging)

**Metadata per specimen:** taxon (species + order), bone type, geological age, specimen ID,
museum catalog number, scanner type, ontogenetic stage, completeness_confidence score.

### Benchmark metrics (standardized)

- Chamfer Distance L1 and L2
- F-Score @ 1% threshold
- Normal Consistency
- Bilateral Symmetry Error (BSE) — novel; reported separately for symmetric vs asymmetric bones
- Uncertainty Calibration: Spearman ρ(per-point std, per-point reconstruction error)

### Release artifacts

- Dataset (CC-BY-4.0 or CC-BY-NC-4.0 per source licenses)
- Evaluation script: `python -m src.eval.evaluate --benchmark paleocomplete`
- Leaderboard on Papers With Code or HuggingFace Datasets

---

## 5. Model design — RELIC

### 5.1 Architecture overview — LION-style two-stage latent diffusion

```
Partial Point Cloud (N×3)
        │
        ▼
┌─────────────────────────────────────┐
│  Confidence-Gated Symmetry Module   │  ← RANSAC plane detection
│  confidence ∈ [0,1]; injects        │     confidence < 0.25 → loss disabled
│  equivariant features into encoder  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Geometry-Aware Encoder             │  ← AdaPoinTr-style point proxies
│  (point proxies + local attention)  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐   ┌──────────────────────────────────┐
│  Hierarchical VAE  [Stage 1]        │   │  Conditioning Encoder            │
│  z_global ∈ ℝ^256                  │   │  (a) Poincaré phylo embedding     │
│  z_local  ∈ ℝ^(K×d)               │   │      (geoopt, dim=64)             │
│  Trained with: CD-L1 + KL          │   │  (b) BioBERT morpho encoder       │
└─────────────────────────────────────┘   │      (fine-tuned on paleo text)  │
        │                          ▲      │  (c) CLIP image encoder           │
        ▼                          │      │      (optional reference bone img)│
┌─────────────────────────────────────┐   └──────────────────────────────────┘
│  DDPM in z_global space  [Stage 2]  │◄── cross-attention on (a)+(b)+(c)
│  T=100 forward, DDIM 20-step sample │    + partial cloud encoder features
│  Null conditioning → CFG at infer. │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  SeedFormer-style Hierarchical      │
│  Transformer Decoder                │
└─────────────────────────────────────┘
        │
        ▼
Complete Point Cloud (M×3) + Normals
```

### 5.2 Loss functions


| Loss                     | Formula                         | Weight          | Notes                             |
| ------------------------ | ------------------------------- | --------------- | --------------------------------- |
| **Chamfer L1**           | `CD(P̂, P*)`                    | 1.0             | Primary reconstruction            |
| **Normal Consistency**   | `1 - cos(n̂, n*)`               | 0.1             | Surface regularity                |
| **Confidence-Gated BSE** | `conf * CD(P̂, reflect(P̂, π))` | 0.1             | Soft; drops to 0 when conf < 0.25 |
| **VAE KL**               | `KL(q(z│x) ∥ p(z))`             | 0.01 (annealed) | Stage 1 latent regularization     |
| **Diffusion ELBO**       | `E[∥ε - ε_θ∥²]`                 | 1.0             | Stage 2 denoising                 |
| **Fidelity**             | `mean_d(P_partial → P̂)`        | 0.5             | Input consistency                 |


### 5.3 Baselines to compare against

1. PointNet AE (simplest deterministic baseline)
2. PoinTr (ICCV 2021 Oral)
3. AdaPoinTr (T-PAMI 2023) — primary deterministic SOTA
4. SeedFormer (ECCV 2022)
5. DiffComplete (NeurIPS 2024) — primary generative SOTA
6. **RELIC** (ours)

All baselines retrained on PaleoComplete using identical data splits and compute budget.
All baselines also evaluated on PCN for cross-benchmark contextualization.

### 5.4 Phylogenetic + morphological conditioning (replaces CLIP text)

**Problem with CLIP:** `"right femur, Tyrannosauridae, Maastrichtian"` is not in CLIP's
training distribution. CLIP has no meaningful understanding of Latin clade names or
stratigraphic epoch terms.

**Solution — two-component conditioning encoder (`src/models/conditioning.py`):**

```python
class PhyloEmbedding(nn.Module):
    """
    Poincaré ball embedding (geoopt) trained on OTL cophenetic distance matrix.
    Query OTL via python-opentree → induced subtree → pairwise distances.
    dim=64, curvature learned. Pre-computed and frozen during RELIC training.
    """

class MorphoBERT(nn.Module):
    """
    BioBERT fine-tuned on MorphoSource project descriptions + paleo literature.
    Understands: bone type, anatomical side, clade names, stratigraphic stage.
    Frozen after fine-tuning; linear projection head on top.
    """

class TaxonomyEncoder(nn.Module):
    """Concatenates PhyloEmbedding + MorphoBERT output → linear → dim=256"""
```

**CLIP image encoder** retained for the optional reference bone image input only.

**At inference without any text/image:** null embedding → classifier-free guidance.

---

## 6. Training and evaluation pipeline

### 6.1 Training schedule


| Stage                             | Data                                      | Epochs | Notes                            |
| --------------------------------- | ----------------------------------------- | ------ | -------------------------------- |
| **Bone pre-train** (VAE only)     | VerSe + BoneDat + ICL (modern bones)      | 100    | Learn bone shape priors; ablated |
| **Fossil domain fine-tune** (VAE) | MorphoSource + Phenome10K fossil meshes   | 50     | Adapt to fossil morphology       |
| **Diffusion training** (Stage 2)  | PaleoComplete train split                 | 30     | Fit latent DDPM                  |
| **DA Phase 2** (adversarial)      | 500–1000 unpaired real MorphoSource scans | 10–15  | Encoder GRL alignment            |
| **DA Phase 3** (self-supervised)  | Same unpaired real scans                  | 5–10   | Masked autoencoding              |


Optimizer: AdamW, lr=1e-4, cosine decay with warmup.
Mixed precision: bfloat16. Batch size: 32 (gradient accumulation if single GPU).
Checkpointing: every 5 epochs; keep top-3 by val CD.

**Estimated compute:** ~4× A100 80GB for full pipeline; documented in paper for reproducibility.

### 6.2 Evaluation suite

**Quantitative (automated):**

- CD-L1, CD-L2 — lower is better
- F-Score @ 1% — higher is better
- Normal Consistency — higher is better
- BSE — lower is better; reported separately for symmetric (femur, tibia, rib) vs
asymmetric/low-confidence specimens
- Diversity: MMD, COV across k=10 completions per partial input
- **Uncertainty calibration**: Spearman ρ between per-point std (across k=10 diffusion
samples) and per-point reconstruction error. Report ρ and p-value. Strong ρ (>0.5)
validates that the uncertainty heatmap is scientifically meaningful.

**Standard CV benchmark:**

- PCN dataset (8 ShapeNet categories) — report in supplementary table to contextualize
RELIC's general completion quality for CV reviewers.

**Qualitative:**

- Open3D renders of predicted vs. ground-truth, colored by per-point error
- Uncertainty heatmap (per-point std, validated by calibration above)
- t-SNE of z_global colored by taxon / bone type (supplementary)

**User study (pre-registered on OSF):**

- **Protocol**: 2AFC forced-choice ("which completion is more plausible, A or B?")
on 20 test specimens; A = RELIC, B = AdaPoinTr (the strongest deterministic baseline)
- **Raters**: minimum 15 domain-matched vertebrate paleontologists (dinosaur specialists
rate only dinosaur bones, etc.); recruited via Society of Vertebrate Paleontology
mailing list
- **Metrics**: win rate vs. baseline, Krippendorff's α inter-rater reliability
- **Pre-registration**: upload protocol to OSF before any data collection

### 6.3 Ablation table


| Model variant           | CD-L1↓ | F@1%↑ | BSE↓ | Calib. ρ↑ | Notes                      |
| ----------------------- | ------ | ----- | ---- | --------- | -------------------------- |
| PointNet AE             | —      | —     | —    | —         | Weakest baseline           |
| AdaPoinTr               | —      | —     | —    | —         | Deterministic SOTA         |
| DiffComplete            | —      | —     | —    | —         | Generative SOTA            |
| RELIC (no pre-train)    | —      | —     | —    | —         | Random init                |
| RELIC (PCN pre-train)   | —      | —     | —    | —         | General shape pre-train    |
| RELIC (bone pre-train)  | —      | —     | —    | —         | Domain pre-train ← current |
| RELIC (no symmetry)     | —      | —     | —    | —         | Ablate symmetry module     |
| RELIC (no phylo cond.)  | —      | —     | —    | —         | Ablate conditioning        |
| RELIC (no domain adapt) | —      | —     | —    | —         | Ablate DA                  |
| **RELIC (full)**        | —      | —     | —    | —         | Proposed method            |


---

## 7. Domain adaptation for fossil scans — three-phase algorithm

### Phase 1 — CT artifact augmentation (during main training, no extra cost)

Stochastic transforms in `src/datasets/augmentations.py` applied to synthetic training data:

- **TPS compression**: thin-plate spline warping along random axis (ratio 0.7–1.0) to
simulate mild taphonomic compression; uses SinPoint-style smooth displacement fields
for topology preservation
- **Affine shear**: random shear matrix (max 0.15) simulating block displacement
- **Ring artifact noise**: concentric cylindrical noise bands
- **Matrix contamination**: random spatial occlusion of point regions (rock matrix)
- **Mineralization infilling**: random interior points in bone cavities
- **Weathering erosion**: surface-normal-directed point deletion on exposed faces
- **Resolution dropout**: random subsampling at variable densities

### Phase 2 — Adversarial encoder alignment (10–15 epochs, unpaired real scans)

```python
# src/domain_adaptation/adaptation.py
# Freeze decoder. Fine-tune encoder only.
# Add domain classifier head on encoder z_global output.
# GRL (gradient reversal layer) trains encoder to be domain-invariant.
# Loss: CE(domain_classifier(z_global), domain_label) reversed via GRL
# Data: 500–1000 unpaired real MorphoSource CT-derived meshes (no ground truth needed)
```

### Phase 3 — Self-supervised masked autoencoding (5–10 epochs, unpaired real scans)

```python
# Randomly mask 40% of real scan → reconstruct via RELIC decoder
# If multiple scans of same specimen available:
#   consistency_loss = CD(RELIC(mask_A(x)), RELIC(mask_B(x)))
# No ground-truth complete shape required at any point
```

---

## 8. Expert-in-the-loop interface

Built with Gradio:

```
Input:
  - Upload partial fossil scan (.ply / .obj / .stl)
  - Optional text: "right tibia, sauropod, Jurassic"  [→ BioBERT + OTL phylo lookup]
  - Optional reference image of related complete bone  [→ CLIP image encoder]

Output panel:
  - Top-5 diverse completions (side-by-side 3D viewer, Open3D WebGL)
  - Per-point uncertainty heatmap (colored by std; validated by Spearman ρ)
  - Symmetry plane overlay + confidence score display
  - Download selected completion as .ply / .stl

Expert actions:
  - Select preferred completion → saved to feedback DB
  - Edit via landmark placement → triggers constrained refinement inference
  - Binary 2AFC rating ("better than baseline?") → feeds user study DB
```

---

## 9. Experiment configuration and reproducibility

All experiments run via:

```bash
python -m src.training.trainer --config experiments/relic_full.yaml
```

Key config fields:

```yaml
model: relic
vae:
  z_global_dim: 256
  z_local_dim: 64
  k_local: 128
diffusion:
  type: ddpm
  T: 100
  ddim_steps: 20
conditioning:
  phylo_embedding: poincare  # poincare | disabled
  phylo_dim: 64
  morpho_encoder: biobert    # biobert | disabled
  image_encoder: clip        # clip | disabled
symmetry:
  enabled: true
  confidence_threshold: 0.25
dataset: paleocomplete
completeness_level: L2       # L1 | L2 | L3 | all
n_points: 4096
batch_size: 32
lr: 1e-4
epochs: 50
seed: 42
```

Logging: Weights & Biases (wandb) with run-level config hash.
Random seeds: fixed globally (torch, numpy, random, cuda).
Preprocessing hash: SHA256 of processed dataset stored in metadata.
GPU: documented A100 count + training wall-clock time for reproducibility.

---

## 10. Related work section (paper outline)

1. **Point cloud completion transformers**: PoinTr [Yu et al., ICCV 2021], AdaPoinTr
  [Yu et al., T-PAMI 2023], SeedFormer [Zhou et al., ECCV 2022], ProxyFormer [Li et al.,
   CVPR 2023]
2. **Diffusion for 3D shape completion**: DiffComplete [Chu et al., NeurIPS 2024], LION
  [Zeng et al., NeurIPS 2022], ShapeFormer [Yan et al., CVPR 2022]
3. **Domain adaptation for shape completion**: SCoDA [Wu et al., CVPR 2023], DAPoinTr
  [2024], RealDiff [arXiv 2409.10180]
4. **Phylogenetic embeddings**: Poincaré embeddings for hierarchies [Nickel & Kiela,
  NeurIPS 2017], hyperbolic phylogenetic tree placement [MDPI 2022], Ph-CNN with
   patristic distances [BMC Bioinformatics 2018]
5. **Biological language models**: BioBERT [Lee et al., Bioinformatics 2020]
6. **Multi-modal 3D completion**: SDS-Complete [2023], P2M2-Net [2024]
7. **Paleontological AI**: Survey [Springer, 2024], CT segmentation for fossils
  [Nature Sci. Rep. 2024]
8. **Biomedical bone reconstruction**: Neural shape completion for maxillofacial surgery
  [Nature Sci. Rep. 2024]

**Positioning**: First work to (a) replace generic text conditioning with domain-appropriate
phylogenetic + morphological embeddings, (b) apply confidence-gated symmetry constraints
specific to bone morphology, (c) formalize a three-phase fossil domain adaptation algorithm,
and (d) evaluate on a purpose-built paleontological benchmark with landmark-based
completeness levels and a pre-registered expert user study.

---

## 11. Future extensions

- **Taphonomically deformed specimens**: extend scope to include geometric correction of
compressed/sheared fossils (separate reconstruction step before completion)
- **Multi-fragment assembly + completion**: multiple disconnected shards → complete bone
- **Mesh-based decoder**: implicit SDF + marching cubes for 3D-printable output
- **Articulated skeleton completion**: extend from single bones to partial skeletons
- **Full phylogenetic priors**: encode continuous evolutionary tree distance as conditioning
(upgrade from discrete taxon lookup to continuous phylogenetic position)
- **Active learning**: paleontologist 2AFC ratings → retraining cycle → iterative improvement
- **Multi-modal inputs**: condition on stratigraphic/taphonomic metadata, 2D photography

