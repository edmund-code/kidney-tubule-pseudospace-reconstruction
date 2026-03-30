# Kidney Tubule Pseudospace Reconstruction

A pipeline for reconstructing the spatial organization of kidney tubule cells from Visium HD spatial transcriptomics data. Tissue is manually segmented into tubule units, cells are classified by nephron segment type, and a pseudospace trajectory is built along the canonical nephron axis — enabling quantitative comparison of gene expression between control and injured kidney tissue.

## Overview

Kidney tubules are continuous structures with distinct cell types arrayed along their length (proximal tubule → thin limb → thick ascending limb → distal nephron). Standard single-cell methods lose this spatial context. This pipeline:

1. Aggregates Visium HD spots into hand-segmented tubule units (GeoJSON polygons)
2. Classifies each unit by nephron segment type using validated marker genes
3. Integrates multiple samples with Harmony batch correction and reconstructs a pseudospace trajectory along the nephron axis
4. Compares gene expression curves between control and ischemia-reperfusion (IR) injury samples to identify spatially-resolved injury signatures

## Samples

| Sample | Species | Condition |
|--------|---------|-----------|
| Ctrl_1A2, Ctrl_1A4 | Mouse | Control |
| IR_2A2, IR_2A4 | Mouse | Ischemia-reperfusion injury (24hr) |
| HUK1_MED, HUK1_COR1 | Human | Control (medulla / cortex) |

## Setup

```bash
conda env create -f environment.yaml
conda activate tubule_pseudospace_reconstruction
```

Additional packages used in notebooks: `harmonypy`, `gseapy`, `seaborn`.

## Pipeline

### Stage 1 — Bin spots to tissue units

Aggregates Visium HD expression spots into per-segment counts based on GeoJSON polygon segmentations.

```bash
python bin2unit.py \
  --geojson segmentations/<sample>.geojson \
  --visium visium_data/<sample>/ \
  --output filtered_tubule_matrices/<sample>.h5ad
```

### Stage 2 — QC and cell type identity assignment

Filters low-quality segments and assigns each to a nephron segment type (PT_S1–S3, DTL1–3, ATL, M_TAL, C_TAL, DCT1–2, CNT, CD_PC). Organism (human/mouse) is auto-detected from gene naming conventions.

```bash
python qc_segmentation.py \
  --input filtered_tubule_matrices/<sample>.h5ad \
  --output processed_data/<sample>_qc.h5ad
```

### Stage 3 — Harmony integration and pseudospace reconstruction

`harmony&analysis.ipynb` — Combines all samples, applies Harmony batch correction, and builds the nephron pseudospace trajectory (PT_S1 → CD_PC). Outputs integrated AnnData to `harmony_integration/`.

### Stage 4 — Within-sample comparison

`within_sample_comparison.ipynb` — Compares gene expression curves along pseudospace between control and IR samples. Computes divergence scores (area between curves) and correlation scores per gene per nephron region. Results saved to `gene_comparison_results/`.

## Key Outputs

- `harmony_integration/` — Harmony-corrected AnnData objects with pseudospace coordinates and cluster assignments
- `gene_comparison_results/` — Per-gene divergence and correlation scores across nephron regions (PT, TAL, DCT, CNT_CD)

## Nephron Trajectory

The pseudospace axis follows the canonical nephron order:

```
PT_S1 → PT_S2 → PT_S3 → DTL1 → DTL2 → DTL3 → ATL → M_TAL → C_TAL → DCT1 → DCT2 → CNT → CD_PC
```

Intercalated cells (CD_IC_A/B) are identified and excluded from the trajectory.
