"""
Bin Gene Expression to Segmented Tissue Units

This script takes a GeoJSON file with segmented tissue units (e.g., RBC clumps)
and a folder with VisiumHD spatial transcriptomics data, then creates a matrix
where each row is a segmented tissue unit and each column is a gene.
Values are the sum of expression from all spots within each tissue unit.
"""

import numpy as np
import pandas as pd
import json
import scanpy as sc
import cv2
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: create a dummy tqdm that just returns the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable


class UnitGeneMatrixCreator:
    """Create tissue unit-by-gene expression matrix from GeoJSON segmentation."""
    
    def __init__(self,
                 geojson_path: str,
                 data_dir: str,
                 scalefactor: Optional[float] = None):
        """
        Initialize the matrix creator.
        
        Args:
            geojson_path: Path to GeoJSON file with segmented tissue units
            data_dir: Directory containing VisiumHD data
            scalefactor: Optional scalefactor override (for coordinate mapping)
        """
        self.geojson_path = Path(geojson_path)
        self.data_dir = Path(data_dir)
        self.scalefactor = scalefactor
        
        # Load data
        self._load_geojson()
        self._load_visium_data()
        self._map_expression_to_units()
    
    def _load_geojson(self):
        """Load GeoJSON file with segmented tissue units."""
        print(f"Loading GeoJSON from {self.geojson_path}...")
        
        with open(self.geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        if geojson_data['type'] != 'FeatureCollection':
            raise ValueError("GeoJSON must be a FeatureCollection")
        
        # Extract polygons from features
        self.units = []
        for feature in geojson_data['features']:
            if feature['geometry']['type'] != 'Polygon':
                print(f"Warning: Skipping non-polygon geometry: {feature['geometry']['type']}")
                continue
            
            # Get coordinates (GeoJSON format: [[[x1, y1], [x2, y2], ...]])
            coords = feature['geometry']['coordinates'][0]  # First ring (exterior)
            
            # Extract unit ID from properties
            unit_id = feature['properties'].get('id', f"unit_{len(self.units)}")
            
            # Convert to numpy array for easier processing
            polygon_points = np.array(coords, dtype=np.float32)
            
            self.units.append({
                'id': unit_id,
                'polygon': polygon_points,
                'properties': feature['properties']
            })
        
        print(f"Loaded {len(self.units)} tissue units from GeoJSON")
    
    def _load_visium_data(self):
        """Load VisiumHD spatial transcriptomics data."""
        print("Loading VisiumHD data...")
        
        # Load scalefactors
        scalefactors_path = self.data_dir / "spatial" / "scalefactors_json.json"
        if scalefactors_path.exists():
            with open(scalefactors_path, 'r') as f:
                self.scalefactors = json.load(f)
        else:
            print("Warning: scalefactors_json.json not found, using default scalefactor=1.0")
            self.scalefactors = {}
        
        # Determine scalefactor to use
        if self.scalefactor is None:
            self.scalefactor = self.scalefactors.get('tissue_hires_scalef', 
                                                      self.scalefactors.get('regist_target_img_scalef', 1.0))
        
        print(f"Using scalefactor: {self.scalefactor}")
        
        # Load tissue positions
        tissue_positions_path = self.data_dir / "spatial" / "tissue_positions.parquet"
        self.tissue_positions = pd.read_parquet(tissue_positions_path)
        
        # Load gene expression matrix
        h5_path = self.data_dir / "filtered_feature_bc_matrix.h5"
        print(f"Loading gene expression from {h5_path}...")
        self.adata = sc.read_10x_h5(h5_path)
        self.adata.var_names_make_unique()
        
        print(f"Gene expression matrix shape: {self.adata.shape}")
        print(f"Number of genes: {len(self.adata.var)}")
        
        # Match barcodes
        expr_barcodes = set(self.adata.obs_names)
        pos_barcodes = set(self.tissue_positions['barcode'])
        common_barcodes = expr_barcodes.intersection(pos_barcodes)
        
        # Filter to common barcodes
        self.adata = self.adata[list(common_barcodes)]
        self.tissue_positions = self.tissue_positions[
            self.tissue_positions['barcode'].isin(common_barcodes)
        ].copy()
        
        # Set index for easier merging
        self.tissue_positions.set_index('barcode', inplace=True)
        self.tissue_positions = self.tissue_positions.reindex(self.adata.obs_names)
        
        # Get all gene names
        self.gene_names = self.adata.var.index.tolist()
        print(f"Loaded {len(self.gene_names)} genes")
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: np.ndarray) -> bool:
        """
        Check if a point is inside a polygon using OpenCV pointPolygonTest.
        
        Args:
            point: (x, y) coordinates
            polygon: Nx2 array of polygon vertices
            
        Returns:
            True if point is inside polygon
        """
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0  # >= 0 means inside or on edge
    
    def _get_polygon_bbox(self, polygon: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Get bounding box of a polygon.
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        return (polygon[:, 0].min(), polygon[:, 1].min(), 
                polygon[:, 0].max(), polygon[:, 1].max())
    
    def _map_expression_to_units(self):
        """Map gene expression spots to tissue units."""
        print("\nMapping gene expression spots to tissue units...")
        
        # Get spot coordinates in full image space
        pxl_col = self.tissue_positions['pxl_col_in_fullres'].values
        pxl_row = self.tissue_positions['pxl_row_in_fullres'].values
        
        # Apply scalefactor to map to image resolution (if needed)
        # GeoJSON coordinates are typically in full resolution
        x_coords = (pxl_col * self.scalefactor).astype(float)
        y_coords = (pxl_row * self.scalefactor).astype(float)
        
        # Filter to in-tissue spots
        in_tissue = self.tissue_positions['in_tissue'].values == 1
        valid_mask = in_tissue
        
        self.valid_x = x_coords[valid_mask]
        self.valid_y = y_coords[valid_mask]
        self.valid_barcodes = self.adata.obs_names[valid_mask]
        
        print(f"Valid spots in tissue: {len(self.valid_x)}")
        
        # Get expression matrix for valid spots
        # Keep sparse format for memory efficiency
        if hasattr(self.adata.X, 'toarray'):
            # It's sparse, keep it sparse
            self.expression_matrix = self.adata.X[valid_mask, :]
            self.is_sparse = True
        else:
            # Already dense
            self.expression_matrix = self.adata.X[valid_mask, :]
            self.is_sparse = False
        
        print(f"Expression matrix shape: {self.expression_matrix.shape}")
        if self.is_sparse:
            print(f"  Sparse matrix format: {type(self.expression_matrix).__name__}")
            print(f"  Non-zero elements: {self.expression_matrix.nnz:,}")
            sparsity = 1.0 - (self.expression_matrix.nnz / (self.expression_matrix.shape[0] * self.expression_matrix.shape[1]))
            print(f"  Sparsity: {sparsity:.2%}")
        
        # Map spots to units
        print("Assigning spots to tissue units...")
        self.spot_to_unit = {}  # Maps spot index to unit index
        
        # Pre-compute bounding boxes for all units (for fast filtering)
        unit_bboxes = []
        for unit in self.units:
            bbox = self._get_polygon_bbox(unit['polygon'])
            unit_bboxes.append(bbox)
        
        # Convert coordinates to numpy arrays for vectorized operations
        spot_x = self.valid_x.values if hasattr(self.valid_x, 'values') else np.array(self.valid_x)
        spot_y = self.valid_y.values if hasattr(self.valid_y, 'values') else np.array(self.valid_y)
        
        # Process units with progress bar
        for unit_idx, unit in enumerate(tqdm(self.units, desc="Mapping spots to units", 
                                             disable=not TQDM_AVAILABLE)):
            polygon = unit['polygon']
            bbox = unit_bboxes[unit_idx]
            min_x, min_y, max_x, max_y = bbox
            
            # First filter by bounding box (much faster than point-in-polygon)
            in_bbox = ((spot_x >= min_x) & (spot_x <= max_x) & 
                      (spot_y >= min_y) & (spot_y <= max_y))
            
            # Only check point-in-polygon for spots within bounding box
            candidate_indices = np.where(in_bbox)[0]
            
            for spot_idx in candidate_indices:
                x, y = spot_x[spot_idx], spot_y[spot_idx]
                if self._point_in_polygon((x, y), polygon):
                    if spot_idx not in self.spot_to_unit:
                        self.spot_to_unit[spot_idx] = []
                    self.spot_to_unit[spot_idx].append(unit_idx)
        
        # Count spots per unit
        unit_spot_counts = {}
        for spot_idx, unit_indices in self.spot_to_unit.items():
            for unit_idx in unit_indices:
                unit_spot_counts[unit_idx] = unit_spot_counts.get(unit_idx, 0) + 1
        
        print(f"Spots assigned to units: {len(self.spot_to_unit)}")
        print(f"Units with spots: {len(unit_spot_counts)}/{len(self.units)}")
        if unit_spot_counts:
            print(f"  Average spots per unit: {np.mean(list(unit_spot_counts.values())):.1f}")
            print(f"  Min spots per unit: {min(unit_spot_counts.values())}")
            print(f"  Max spots per unit: {max(unit_spot_counts.values())}")
    
    def create_unit_gene_matrix(self) -> Tuple[pd.DataFrame, sparse.csr_matrix]:
        """
        Create tissue unit-by-gene expression matrix.
        Uses sparse matrix format for memory efficiency.
        
        Returns:
            Tuple of (metadata_df, expression_matrix_sparse)
            - metadata_df: DataFrame with unit_id, x_centroid, y_centroid, n_spots
            - expression_matrix_sparse: Sparse matrix (units x genes)
        """
        print("\nCreating tissue unit-by-gene matrix...")
        
        # Pre-build reverse mapping: unit_idx -> list of spot_indices
        unit_to_spots = {i: [] for i in range(len(self.units))}
        for spot_idx, unit_indices in self.spot_to_unit.items():
            for unit_idx in unit_indices:
                unit_to_spots[unit_idx].append(spot_idx)
        
        # Prepare metadata
        metadata_rows = []
        expression_rows = []
        
        for unit_idx, unit in enumerate(tqdm(self.units, desc="Creating expression matrix",
                                            disable=not TQDM_AVAILABLE)):
            unit_id = unit['id']
            polygon = unit['polygon']
            
            # Calculate centroid
            centroid_x = float(np.mean(polygon[:, 0]))
            centroid_y = float(np.mean(polygon[:, 1]))
            
            # Get spots for this unit (from pre-built mapping)
            spot_indices = unit_to_spots[unit_idx]
            
            # Sum expression for all spots within this unit
            if len(spot_indices) > 0:
                # Use sparse matrix operations
                if self.is_sparse:
                    # Extract rows and sum (keeps sparse format)
                    unit_expression = self.expression_matrix[spot_indices, :].sum(axis=0)
                    # Convert to 1D array (still sparse if possible)
                    if hasattr(unit_expression, 'A1'):
                        unit_expression = unit_expression.A1  # Convert sparse to dense 1D array
                    elif hasattr(unit_expression, 'toarray'):
                        unit_expression = unit_expression.toarray().flatten()
                    else:
                        unit_expression = np.array(unit_expression).flatten()
                else:
                    unit_expression = self.expression_matrix[spot_indices, :].sum(axis=0)
            else:
                # No spots in this unit
                unit_expression = np.zeros(self.expression_matrix.shape[1])
            
            # Store metadata
            metadata_rows.append({
                'unit_id': unit_id,
                'x_centroid': centroid_x,
                'y_centroid': centroid_y,
                'n_spots': len(spot_indices)
            })
            
            # Store expression (as list for now, will convert to sparse matrix)
            expression_rows.append(unit_expression)
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_rows)
        
        # Convert expression rows to sparse matrix
        print("Converting to sparse matrix format...")
        with tqdm(total=1, desc="Building sparse matrix", disable=not TQDM_AVAILABLE) as pbar:
            expression_dense = np.array(expression_rows, dtype=np.float32)  # Use float32 to save memory
            expression_sparse = sparse.csr_matrix(expression_dense)
            pbar.update(1)
        
        print(f"\nCreated matrix with {len(metadata_df)} units and {len(self.gene_names)} genes")
        print(f"Expression matrix shape: {expression_sparse.shape}")
        print(f"  Non-zero elements: {expression_sparse.nnz:,}")
        sparsity = 1.0 - (expression_sparse.nnz / (expression_sparse.shape[0] * expression_sparse.shape[1]))
        print(f"  Sparsity: {sparsity:.2%}")
        print(f"Units with spots: {(metadata_df['n_spots'] > 0).sum()}/{len(metadata_df)}")
        
        return metadata_df, expression_sparse
    
    def visualize_unit_expression(self,
                                  df: pd.DataFrame,
                                  gene_names: List[str],
                                  output_path: str,
                                  histology_image_path: Optional[str] = None):
        """
        Visualize unit expression for specified genes.
        Note: df should have unit_id, x_centroid, y_centroid, and gene columns.
        """
        """
        Visualize unit expression for specified genes.
        
        Args:
            df: Unit-by-gene DataFrame
            gene_names: List of gene names to visualize
            output_path: Path to save visualization
            histology_image_path: Optional path to histology image for background
        """
        print(f"\nCreating visualization for genes: {gene_names}")
        
        n_genes = len(gene_names)
        fig, axes = plt.subplots(1, n_genes, figsize=(12 * n_genes, 12))
        
        if n_genes == 1:
            axes = [axes]
        
        for gene_idx, gene_name in enumerate(gene_names):
            ax = axes[gene_idx]
            
            # Find gene (case-insensitive, handle suffixes)
            gene_col = None
            gene_upper = gene_name.upper()
            
            # Try exact match first
            if gene_name in df.columns:
                gene_col = gene_name
            else:
                # Try case-insensitive
                for col in df.columns:
                    if col.upper() == gene_upper:
                        gene_col = col
                        break
                
                # Try with common suffixes
                if gene_col is None:
                    for suffix in ['-1', '-2', '-3']:
                        for col in df.columns:
                            if col.upper() == (gene_upper + suffix):
                                gene_col = col
                                break
                        if gene_col:
                            break
            
            if gene_col is None:
                print(f"Warning: Gene {gene_name} not found in matrix")
                # Try to find similar
                similar = [c for c in df.columns if gene_upper in c.upper()][:5]
                if similar:
                    print(f"  Similar gene names: {similar}")
                ax.text(0.5, 0.5, f"Gene {gene_name} not found", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            expression_values = df[gene_col].values
            x_coords = df['x_centroid'].values
            y_coords = df['y_centroid'].values
            
            # Filter to units with expression
            has_expression = expression_values > 0
            
            if has_expression.sum() == 0:
                ax.text(0.5, 0.5, f"No expression detected for {gene_col}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Create scatter plot
            scatter = ax.scatter(
                x_coords[has_expression], y_coords[has_expression],
                c=expression_values[has_expression],
                s=100,
                cmap='YlOrRd',
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'{gene_col} Expression (sum)', rotation=270, labelpad=20)
            
            ax.set_title(f'{gene_name} ({gene_col}) Unit Expression\n'
                        f'{len(df)} units, {has_expression.sum()} with expression',
                        fontsize=14)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create tissue unit-by-gene expression matrix from GeoJSON segmentation'
    )
    parser.add_argument('--geojson', '-g', type=str, required=True,
                       help='Path to GeoJSON file with segmented tissue units')
    parser.add_argument('--data-dir', '-d', type=str, required=True,
                       help='Directory containing VisiumHD data (e.g., square_002um)')
    parser.add_argument('--output', '-o', type=str, default='unit_gene_matrix.npz',
                       help='Output path for matrix file (default: unit_gene_matrix.npz). '
                            'Use .npz for sparse format, .csv.gz for dense CSV, or .h5ad for AnnData format.')
    parser.add_argument('--scalefactor', type=float, default=None,
                       help='Optional scalefactor override for coordinate mapping')
    parser.add_argument('--validate-genes', type=str, nargs='+', default=[],
                       help='Genes to visualize for validation (e.g., Slc5a2 Nphs2)')
    parser.add_argument('--validation-output', type=str, default='unit_expression_validation.png',
                       help='Output path for validation visualization')
    
    args = parser.parse_args()
    
    # Create matrix creator
    creator = UnitGeneMatrixCreator(
        geojson_path=args.geojson,
        data_dir=args.data_dir,
        scalefactor=args.scalefactor
    )
    
    # Create matrix
    metadata_df, expression_sparse = creator.create_unit_gene_matrix()
    
    # Save based on file extension
    output_path = Path(args.output)
    print(f"\nSaving matrix to {args.output}...")
    
    if output_path.suffix == '.npz':
        # Save as sparse .npz format (most efficient for sparse data)
        print("Saving as sparse .npz format...")
        with tqdm(total=1, desc="Writing .npz file", disable=not TQDM_AVAILABLE) as pbar:
            # Save sparse matrix components separately
            np.savez_compressed(
                args.output,
                # Sparse matrix components
                expression_data=expression_sparse.data,
                expression_indices=expression_sparse.indices,
                expression_indptr=expression_sparse.indptr,
                expression_shape=expression_sparse.shape,
                # Metadata
                unit_ids=metadata_df['unit_id'].values,
                x_centroids=metadata_df['x_centroid'].values,
                y_centroids=metadata_df['y_centroid'].values,
                n_spots=metadata_df['n_spots'].values,
                gene_names=np.array(creator.gene_names)
            )
            pbar.update(1)
        print(f"Saved sparse matrix: {expression_sparse.shape[0]} units x {expression_sparse.shape[1]} genes")
        print(f"  File size: {Path(args.output).stat().st_size / 1024 / 1024:.2f} MB")
        print(f"\nTo load this file:")
        print(f"  data = np.load('{args.output}', allow_pickle=True)")
        print(f"  expression = sparse.csr_matrix((data['expression_data'], data['expression_indices'],")
        print(f"                                  data['expression_indptr']), shape=data['expression_shape'])")
        print(f"  unit_ids = data['unit_ids']")
        print(f"  gene_names = data['gene_names']")
        
    elif output_path.suffix == '.h5ad':
        # Save as AnnData format (compatible with scanpy/anndata)
        print("Saving as AnnData (.h5ad) format...")
        import anndata as ad
        
        # Create AnnData object
        adata = ad.AnnData(X=expression_sparse)
        adata.obs = metadata_df.set_index('unit_id')
        adata.var = pd.DataFrame(index=creator.gene_names)
        adata.var_names = creator.gene_names
        
        adata.write(args.output)
        print(f"Saved AnnData: {adata.shape[0]} units x {adata.shape[1]} genes")
        print(f"  File size: {Path(args.output).stat().st_size / 1024 / 1024:.2f} MB")
        
    elif output_path.suffixes == ['.csv', '.gz'] or output_path.suffix == '.csv':
        # Save as CSV (dense format - may be slow/large for sparse data)
        print("Saving as CSV format (converting to dense - this may take a while for sparse data)...")
        print("  Warning: CSV format is inefficient for sparse data. Consider using .npz or .h5ad instead.")
        
        # Convert sparse to dense for CSV
        with tqdm(total=1, desc="Converting to dense", disable=not TQDM_AVAILABLE) as pbar:
            expression_dense = expression_sparse.toarray()
            pbar.update(1)
        
        # Create full DataFrame
        expression_df = pd.DataFrame(
            expression_dense,
            columns=creator.gene_names,
            index=metadata_df['unit_id']
        )
        
        # Combine with metadata
        df = pd.concat([metadata_df.set_index('unit_id'), expression_df], axis=1)
        df = df.reset_index()
        
        # Save
        compression = 'gzip' if output_path.suffix == '.gz' else None
        df.to_csv(args.output, index=False, compression=compression)
        print(f"Saved CSV: {len(df)} units x {len(df.columns) - 4} genes")
        print(f"  File size: {Path(args.output).stat().st_size / 1024 / 1024:.2f} MB")
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}. Use .npz, .h5ad, or .csv.gz")
    
    # Create validation visualization if genes specified
    if args.validate_genes:
        print(f"\nCreating validation visualization...")
        # Create temporary DataFrame for visualization
        expression_dense = expression_sparse.toarray()
        expression_df = pd.DataFrame(
            expression_dense,
            columns=creator.gene_names,
            index=metadata_df['unit_id']
        )
        df_viz = pd.concat([metadata_df.set_index('unit_id'), expression_df], axis=1).reset_index()
        
        creator.visualize_unit_expression(
            df=df_viz,
            gene_names=args.validate_genes,
            output_path=args.validation_output
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()