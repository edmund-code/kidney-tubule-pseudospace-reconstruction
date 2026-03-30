#!/usr/bin/env python3
"""
qc_control.py

Minimal QC pipeline (fast, no PCA/HVG/UMAP) that:
 - Loads AnnData (.h5ad)
 - Filters segments by min_counts / min_genes (hard-coded: 200 / 50)
 - Recalculates validated region markers (>=10% detection) from full marker lists
 - Computes identity scores (per-marker z-score, averaged per region)
 - Computes entropy-based coherence, assigns primary/secondary identities
 - Flags 'ok' / 'weak_identity' / 'incompatible'
 - Writes only the final filtered AnnData (.h5ad) with identity_flag == 'ok'

Usage:
    python qc_control.py --input /path/to/adata.h5ad --organism human --output /path/to/filtered.h5ad

Notes:
 - Requires: scanpy, numpy, pandas, scipy
 - For human, uses the Lake et al. Supplementary Table 5 gene lists (uppercase).
 - For mouse, uses the mouse lists (mixed case) provided earlier.
 - The script recalculates validated markers each run (>=10% expressed).
"""

import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata

# ---------------------------
# Full marker dictionaries
# ---------------------------

# Human markers (Lake et al. Supplementary Table 5)
tubule_markers_human = {
    'PT_S1': ['SLC5A12', 'SLC13A3', 'SLC22A6', 'PRODH2', 'SLC5A2', 'SLC22A8'],
    'PT_S2': ['SLC5A12', 'SLC13A3', 'SLC22A6', 'SLC34A1', 'SLC22A7'],
    'PT_S3': ['SLC22A7', 'MOGAT1', 'SLC5A11', 'SLC22A24', 'SLC7A13', 'SLC5A8', 'ABCC3', 'SATB2'],
    'DTL1': ['SATB2', 'JAG1', 'ADGRL3', 'ID1'],
    'DTL2': ['VCAM1', 'SLC39A8', 'AQP1', 'LRRC4C', 'LRP2', 'UNC5D', 'SATB2'],
    'DTL3': ['CLDN1', 'AKR1B1', 'CLDN4', 'BCL6', 'SH3GL3', 'SLC14A2', 'SMOC2'],
    'ATL': ['CLDN1', 'AKR1B1', 'CLDN4', 'BCL6', 'SH3GL3', 'BCAS1', 'CLCNKA', 'CLDN10', 'PROX1'],
    'M_TAL': ['NELL1', 'ESRRB', 'EGF', 'CLDN14', 'PROX1', 'MFSD4A', 'KCTD16', 'RAP1GAP', 'ANK2', 'CYFIP2', 'SLC12A1', 'UMOD'],
    'C_TAL': ['NELL1', 'ESRRB', 'EGF', 'PPM1E', 'GP2', 'ENOX1', 'TMEM207', 'TMEM52B', 'CLDN16', 'WNK1', 'SLC12A1', 'UMOD'],
    'DCT1': ['TRPM7', 'ADAMTS17', 'ITPKB', 'ZNF385D', 'HS6ST2', 'SLC12A3', 'TRPM6'],
    'DCT2': ['TRPV5', 'SLC8A1', 'SCN2A', 'HSD11B2', 'CALB1', 'SLC12A3'],
    'CNT': ['SLC8A1', 'SCN2A', 'HSD11B2', 'CALB1', 'KITLG', 'PCDH7'],
    'CD_PC': ['AQP2', 'AQP3', 'SCNN1G', 'SCNN1B', 'FXYD4', 'GATA3'],
    'CD_IC_A': ['SLC4A1', 'SLC26A7', 'ATP6V0D2', 'ATP6V1C2', 'TMEM213', 'CLNK'],
    'CD_IC_B': ['SLC4A9', 'SLC35F3', 'SLC26A4', 'INSRR', 'TLDC2'],
}

major_region_markers_human = {
    'Proximal_Tubule': ['LRP2', 'CUBN', 'SLC5A2', 'SLC5A12', 'SLC13A3', 'SLC22A6', 'SLC34A1'],
    'Thin_Limb': ['AQP1', 'CLDN10', 'CRYAB', 'TACSTD2', 'CLDN1'],
    'TAL': ['SLC12A1', 'UMOD', 'CLDN16', 'EGF', 'CASR'],
    'DCT': ['SLC12A3', 'TRPM6', 'CNNM2', 'PVALB'],
    'CNT_CD': ['CALB1', 'AQP2', 'AQP3', 'SCNN1B', 'SLC4A1', 'SLC26A4'],
}

# Mouse full lists (from the notebook)
tubule_markers_mouse = {
    'PT_S1': ['Slc5a12', 'Slc13a3', 'Slc22a6', 'Prodh2', 'Slc5a2', 'Slc22a8'],
    'PT_S2': ['Slc5a12', 'Slc13a3', 'Slc22a6', 'Slc34a1', 'Slc22a7'],
    'PT_S3': ['Slc22a7', 'Mogat1', 'Slc5a11', 'Slc22a24', 'Slc7a13', 'Slc5a8', 'Abcc3', 'Satb2'],
    'DTL1': ['Satb2', 'Jag1', 'Adgrl3', 'Id1'],
    'DTL2': ['Vcam1', 'Slc39a8', 'Aqp1', 'Lrrc4c', 'Lrp2', 'Unc5d', 'Satb2'],
    'DTL3': ['Cldn1', 'Akr1b3', 'Cldn4', 'Bcl6', 'Sh3gl3', 'Slc14a2', 'Smoc2'],
    'ATL': ['Cldn1', 'Akr1b3', 'Cldn4', 'Bcl6', 'Sh3gl3', 'Bcas1', 'Clcnka', 'Cldn10', 'Prox1'],
    'M_TAL': ['Nell1', 'Esrrb', 'Egf', 'Cldn14', 'Prox1', 'Mfsd4a', 'Kctd16', 'Rap1gap', 'Ank2', 'Cyfip2', 'Slc12a1', 'Umod'],
    'C_TAL': ['Nell1', 'Esrrb', 'Egf', 'Ppm1e', 'Gp2', 'Enox1', 'Tmem207', 'Tmem52b', 'Cldn16', 'Wnk1', 'Slc12a1', 'Umod'],
    'DCT1': ['Trpm7', 'Adamts17', 'Itpkb', 'Znf385d', 'Hs6st2', 'Slc12a3', 'Trpm6'],
    'DCT2': ['Trpv5', 'Slc8a1', 'Scn2a', 'Hsd11b2', 'Calb1', 'Slc12a3'],
    'CNT': ['Slc8a1', 'Scn2a', 'Hsd11b2', 'Calb1', 'Kitlg', 'Pcdh7'],
    'CD_PC': ['Aqp2', 'Aqp3', 'Scnn1g', 'Scnn1b', 'Fxyd4', 'Gata3'],
    'CD_IC_A': ['Slc4a1', 'Slc26a7', 'Atp6v0d2', 'Atp6v1c2', 'Tmem213', 'Clnk'],
    'CD_IC_B': ['Slc4a9', 'Slc35f3', 'Slc26a4', 'Insrr', 'Tldc2'],
}

major_region_markers_mouse = {
    'Proximal_Tubule': ['Lrp2', 'Cubn', 'Slc5a2', 'Slc5a12', 'Slc13a3', 'Slc22a6', 'Slc34a1'],
    'Thin_Limb': ['Aqp1', 'Cldn10', 'Cryab', 'Tacstd2', 'Cldn1'],
    'TAL': ['Slc12a1', 'Umod', 'Cldn16', 'Egf', 'Casr'],
    'DCT': ['Slc12a3', 'Trpm6', 'Cnnm2', 'Pvalb'],
    'CNT_CD': ['Calb1', 'Aqp2', 'Aqp3', 'Scnn1b', 'Slc4a1', 'Slc26a4'],
}

# ---------------------------
# Helper utilities
# ---------------------------

def get_expr_and_genes(adata):
    """Return expression matrix (numpy array) and gene list from adata.raw if present else adata."""
    if adata.raw is not None:
        expr = adata.raw.X
        genes = list(adata.raw.var_names)
    else:
        expr = adata.X
        genes = list(adata.var_names)
    if hasattr(expr, 'toarray'):
        expr = expr.toarray()
    return expr, genes

def profile_markers_detection(adata, marker_dict, min_pct=10.0):
    """
    For each marker in marker_dict, compute percent of segments expressing (>0).
    Return validated markers per region (pct >= min_pct) and per-gene pct map.
    """
    expr, genes = get_expr_and_genes(adata)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n_segments = expr.shape[0]
    validated = {}
    pct_map = {}
    all_markers = sorted({g for gl in marker_dict.values() for g in gl})
    for g in all_markers:
        if g not in gene_to_idx:
            pct_map[g] = 0.0
        else:
            col = expr[:, gene_to_idx[g]]
            pct_map[g] = 100.0 * np.sum(col > 0) / n_segments
    for seg, genes_list in marker_dict.items():
        good = [g for g in genes_list if pct_map.get(g, 0.0) >= min_pct]
        validated[seg] = good
    return validated, pct_map

def calculate_identity_scores(adata, marker_dict, method='zscore'):
    expr, genes = get_expr_and_genes(adata)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n_segments = expr.shape[0]
    scores = {}
    for seg, markers in marker_dict.items():
        valid = [m for m in markers if m in gene_to_idx]
        if not valid:
            scores[seg] = np.zeros(n_segments)
            continue
        idx = [gene_to_idx[m] for m in valid]
        mat = expr[:, idx]
        if method == 'zscore':
            z = np.zeros_like(mat)
            for j in range(mat.shape[1]):
                col = mat[:, j]
                s = col.std()
                if s > 0:
                    z[:, j] = (col - col.mean()) / s
                else:
                    z[:, j] = 0.0
            scores[seg] = z.mean(axis=1)
        elif method == 'percentile':
            pct = np.zeros_like(mat)
            for j in range(mat.shape[1]):
                pct[:, j] = rankdata(mat[:, j]) / n_segments
            scores[seg] = pct.mean(axis=1)
        else:
            scores[seg] = mat.mean(axis=1)
    return pd.DataFrame(scores, index=adata.obs_names)

def calculate_identity_coherence(scores_df):
    # Entropy-based coherence: 1 - normalized entropy
    scores_shifted = scores_df.sub(scores_df.min(axis=1), axis=0)
    ssum = scores_shifted.sum(axis=1)
    probs = scores_shifted.div(ssum, axis=0).fillna(1.0 / scores_df.shape[1])
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    max_entropy = np.log(scores_df.shape[1])
    coherence = 1 - (entropy / max_entropy)
    return coherence

def assign_identity(scores_df, coherence, min_coherence=0.3):
    top2 = scores_df.apply(lambda x: x.nlargest(2).index.tolist(), axis=1)
    top2_vals = scores_df.apply(lambda x: x.nlargest(2).values.tolist(), axis=1)
    df = pd.DataFrame({
        'primary_identity': [t[0] for t in top2],
        'secondary_identity': [t[1] if len(t) > 1 else None for t in top2],
        'primary_score': [v[0] for v in top2_vals],
        'secondary_score': [v[1] if len(v) > 1 else 0 for v in top2_vals],
        'coherence': coherence,
    }, index=scores_df.index)
    df['confident'] = df['coherence'] >= min_coherence
    return df

def flag_incompatible_segments(region_scores, threshold=0.5):
    compatible_neighbors = {
        'Proximal_Tubule': ['Thin_Limb'],
        'Thin_Limb': ['Proximal_Tubule', 'TAL'],
        'TAL': ['Thin_Limb', 'DCT'],
        'DCT': ['TAL', 'CNT_CD'],
        'CNT_CD': ['DCT'],
    }
    flags = []
    for idx in region_scores.index:
        scores = region_scores.loc[idx]
        top2 = scores.nlargest(2)
        primary = top2.index[0]
        secondary = top2.index[1]
        compatible = secondary in compatible_neighbors.get(primary, []) or secondary == primary
        both_high = (top2.values[0] > threshold) and (top2.values[1] > threshold)
        if both_high and not compatible:
            flags.append('incompatible')
        elif top2.values[0] < 0:
            flags.append('weak_identity')
        else:
            flags.append('ok')
    return pd.Series(flags, index=region_scores.index)

# ---------------------------
# Main (minimal) pipeline
# ---------------------------

def main(args):
    # Hard-coded thresholds
    min_counts = 200
    min_genes = 50
    coherence_threshold = 0.3
    flag_threshold = 0.5
    marker_validation_pct = 10.0

    # Load AnnData
    adata = sc.read_h5ad(args.input)

    # Remove segments with zero spots if available
    if 'n_spots' in adata.obs.columns:
        adata = adata[adata.obs['n_spots'] > 0].copy()

    # Basic QC metrics and filtering
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['total_counts'] >= min_counts].copy()
    adata = adata[adata.obs['n_genes_by_counts'] >= min_genes].copy()

    # Ensure expression source: use adata.raw if available, else adata.X
    # Store counts layer for safety
    adata.layers['counts'] = adata.X.copy()
    # Standard normalization and log transform (keeps same pipeline as notebook)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # use normalized logged data for scoring

    # Select major region markers according to organism
    if args.organism == 'human':
        major_region_markers = major_region_markers_human
        all_markers_dict = tubule_markers_human
    else:
        major_region_markers = major_region_markers_mouse
        all_markers_dict = tubule_markers_mouse

    # Recalculate validated markers using >=10% expression
    validated_region_markers, pct_map = profile_markers_detection(adata, major_region_markers, min_pct=marker_validation_pct)

    # Warn about regions with few or no validated markers
    for region, markers in validated_region_markers.items():
        if len(markers) == 0:
            print(f"WARNING: region '{region}' has 0 validated markers (>= {marker_validation_pct}% expr).")
        elif len(markers) < 2:
            print(f"WARNING: region '{region}' has only {len(markers)} validated marker(s): {markers}")

    # Compute identity scores (z-score per marker, averaged)
    region_scores = calculate_identity_scores(adata, validated_region_markers, method='zscore')

    # Compute coherence (entropy-based)
    coherence = calculate_identity_coherence(region_scores)

    # Assign identities
    identity_df = assign_identity(region_scores, coherence, min_coherence=coherence_threshold)

    # Add identity info to adata.obs
    adata.obs['primary_identity'] = identity_df['primary_identity'].astype(str)
    adata.obs['secondary_identity'] = identity_df['secondary_identity'].astype(str)
    adata.obs['identity_coherence'] = identity_df['coherence'].values
    adata.obs['identity_confident'] = identity_df['confident'].values
    for col in region_scores.columns:
        adata.obs[f'score_{col}'] = region_scores[col].values

    # Flag incompatible / weak identities
    flags = flag_incompatible_segments(region_scores, threshold=flag_threshold)
    adata.obs['identity_flag'] = flags

    # Final filtering: keep only identity_flag == 'ok'
    keep_mask = adata.obs['identity_flag'] == 'ok'
    adata_filtered = adata[keep_mask].copy()

    # Save final filtered h5ad only
    adata_filtered.write(args.output)
    print(f"Saved filtered AnnData to: {args.output}")
    print(f"Retained {adata_filtered.n_obs} / {adata.n_obs} segments ({100*adata_filtered.n_obs/adata.n_obs:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal QC pipeline (no PCA/HVG/UMAP)")
    parser.add_argument('--input', required=True, help="Input AnnData .h5ad file")
    parser.add_argument('--output', required=True, help="Output filtered AnnData .h5ad file")
    parser.add_argument('--organism', choices=['mouse','human'], default='human', help="Organism (mouse or human)")
    args = parser.parse_args()
    main(args)