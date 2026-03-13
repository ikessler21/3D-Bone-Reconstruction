# Plan: finetune_fossil.yaml + construct_paleocomplete.py

## Context

The RELIC training pipeline has three sequential stages:
1. **Bone pre-train** (VAE only, proxy data) → `pretrain_bone.yaml` ✅ done
2. **Fossil fine-tune** (VAE only, real fossil data) → `finetune_fossil.yaml` ← **this plan**
3. **Diffusion training** (PaleoComplete) → needs PaleoComplete benchmark first

Two missing pieces block stages 2 and 3:
- `experiments/finetune_fossil.yaml` — config for fossil fine-tuning
- `scripts/construct_paleocomplete.py` — builds the benchmark from preprocessed Phenome10K .pt files

---

## File 1: `experiments/finetune_fossil.yaml`

**Template:** `experiments/pretrain_bone.yaml` (same key names, same mode: `--mode pretrain`)

**Key differences from pretrain_bone.yaml:**

| Parameter | pretrain_bone | finetune_fossil | Reason |
|---|---|---|---|
| `data.processed_dir` | `data/processed` | `data/processed/phenome10k` | Isolate fossil distribution; avoid dilution with modern bones |
| `epochs` | 100 | 50 | Fine-tuning from converged checkpoint |
| `training.lr` | 1e-4 | 5e-5 | ~5× lower: fine-tuning, not learning from scratch |
| `training.warmup_epochs` | 5 | 2 | Model already in a good basin |
| `batch_size` | 16 | 24 | Phenome10K (~1,400 specimens) is larger; no need to accumulate |
| `grad_accum_steps` | 2 | 1 | Effective batch 24 (vs 32 in pretrain; acceptable) |
| `loss_weights.symmetry` | 0.1 | 0.15 | Real fossils have genuine bilateral symmetry; confidence gate protects asymmetric specimens |
| `loss_weights.diffusion` | 0.0 | 0.0 | Still `--mode pretrain` (diffusion frozen) |
| `beta_warmup` | 2000 | 1500 | 50 epochs × ~58 steps ≈ 2,900 total steps; beta_max reached at epoch ~26 |
| `checkpoint_dir` | `checkpoints/pretrain_bone` | `checkpoints/finetune_fossil` | |
| `run_name` | `pretrain_bone` | `finetune_fossil` | |

**Usage comment block (in the YAML header):**
```
# Run:
#   python -m src.training.trainer \
#     --config experiments/finetune_fossil.yaml --mode pretrain \
#     --checkpoint checkpoints/pretrain_bone/best_epoch<N>_cd<X>.pth
```

---

## File 2: `scripts/construct_paleocomplete.py`

### What it does

Converts preprocessed Phenome10K `.pt` files into the PaleoComplete benchmark:
- Generates L1/L2/L3 partial versions of each specimen
- Assigns Tier A (conf ≥ 0.8) → test only; Tier B (0.5–0.8) → train/val + small test fraction
- Writes `data/paleocomplete/{train,val,test}/*.pt` + three split JSONs

### Key design decisions

**Proxy landmarks (no anatomical annotations available):** Use the 6 PCA bounding-box extrema (±x, ±y, ±z in the already-normalized PCA frame) as proxy landmark positions. These are deterministic, fast (O(N)), and morphologically interpretable (proximal/distal, medial/lateral, anterior/posterior extremes).

**Reuse `LandmarkShardGenerator`** (already in `src/datasets/augmentations.py`) — instantiate it with the 6 computed proxy landmarks and call `.generate(full_pts, level)`. This reuses the exact logic used in training augmentation and ensures consistent L1/L2/L3 semantics: `LEVEL_MISSING = {"L1":(1,2), "L2":(3,3), "L3":(4,6)}`.

**Split strategy:**
- All Tier A (conf ≥ 0.8) → test split exclusively
- Tier B: stratify by `bone_type` → 70% train / 15% val / 15% test
- Combined test = Tier A + 15% Tier B

**Idempotent:** `--resume` reloads existing split JSONs (to preserve assignments) and skips already-written `.pt` files.

### Algorithm

```
1. discover_specimens(source_dir, min_confidence=0.5)
     → load each .pt, read metadata, tier = "A" if conf >= 0.8 else "B"

2. if --resume and all 3 split JSONs exist:
       load existing split assignment
   else:
       split_specimens(records, train_frac=0.70, val_frac=0.15, seed)
         → Tier A → test; Tier B stratified by bone_type

3. for each split in [train, val, test]:
     for each specimen in split:
       landmarks = compute_proxy_landmarks(full_pts)
         → {"pos_x": Tensor[3], "neg_x": ..., "pos_y", "neg_y", "pos_z", "neg_z"}
       shard_gen = LandmarkShardGenerator(landmark_centers=landmarks, radius=0.20)
       for level in ["L1", "L2", "L3"]:
         out_path = out_dir / split / f"{specimen_id}_{level}.pt"
         if --resume and out_path.exists(): skip generation
         else:
           partial, removed_names = shard_gen.generate(full_pts, level)
           if partial.shape[0] < 64: fallback to make_partial()
           torch.save({full, partial, metadata + {completeness_level, fraction_removed}}, out_path)

4. write {split}_split.json for each split
     pt_path is relative, forward slashes (for cross-platform)
```

### Output structure

```
data/paleocomplete/
├── train/  slug1_L1.pt, slug1_L2.pt, slug1_L3.pt, ...
├── val/    ...
├── test/   ...  (all Tier A here)
├── train_split.json
├── val_split.json
└── test_split.json
```

**Split JSON record format** (matches `PaleoCompleteDataset._load_from_split_json`):
```json
{
  "id": "slug_L2",
  "pt_path": "train/slug_L2.pt",
  "completeness_level": "L2",
  "bone_type": "unknown",
  "taxon": "...",
  "completeness_confidence": 0.6,
  "fraction_removed": 0.47,
  "specimen_id": "slug",
  "geological_age": "unknown",
  "taxon_order": "unknown"
}
```

### CLI

```bash
python scripts/construct_paleocomplete.py
python scripts/construct_paleocomplete.py --source-dir data/processed/phenome10k --out-dir data/paleocomplete
python scripts/construct_paleocomplete.py --resume
python scripts/construct_paleocomplete.py --radius 0.20 --workers 8 --seed 42
```

---

## Critical files

| File | Role |
|---|---|
| `experiments/pretrain_bone.yaml` | Direct template — all YAML key names must match |
| `src/datasets/augmentations.py` | `LandmarkShardGenerator` + `LEVEL_MISSING` (lines 42–110) to reuse directly |
| `src/utils/geometry.py` | `landmark_region_mask` (line 202) — used internally by `LandmarkShardGenerator` |
| `src/datasets/fossil_dataset.py` | `_load_from_split_json` (line 221) — defines required JSON fields and `pt_path` format |
| `scripts/preprocess_phenome10k.py` | Defines the source `.pt` metadata schema (all specimens have `completeness_confidence: 0.6`) |
| `scripts/download_and_preprocess.py` | `make_partial()` fallback for degenerate specimens |

---

## Verification

After implementation:

1. **finetune_fossil.yaml smoke test:**
   ```bash
   python -m src.training.trainer --config experiments/finetune_fossil.yaml --mode pretrain
   # Should start training (will error if no processed/phenome10k data yet; that's expected)
   ```

2. **PaleoComplete construction test:**
   ```bash
   # With at least a few Phenome10K .pt files present:
   python scripts/construct_paleocomplete.py --source-dir data/processed/phenome10k
   # Check:
   python -c "import json; d=json.load(open('data/paleocomplete/train_split.json')); print(len(d), d[0].keys())"
   # Should print record count and field names including 'completeness_level', 'pt_path'
   ```

3. **Dataset loading test:**
   ```python
   from src.datasets.fossil_dataset import PaleoCompleteDataset
   ds = PaleoCompleteDataset("data/paleocomplete", split="train")
   print(len(ds), ds[0]["partial"].shape, ds[0]["metadata"]["completeness_level"])
   ```

4. **Diffusion training startup:**
   ```bash
   python -m src.training.trainer --config experiments/relic_full.yaml --mode diffusion \
     --checkpoint checkpoints/finetune_fossil/best_epoch050_cd<X>.pth
   ```
