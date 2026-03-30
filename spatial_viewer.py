#!/usr/bin/env python3
"""
Spatial Transcriptomics Viewer
================================
Interactive Dash/Plotly web app for visualising Visium HD spatial transcriptomics
data overlaid on H&E histology images. Modelled on Cartoscope.

Data Input Modes
----------------
Mode A — Raw 10x Space Ranger output directory:

    <sample_dir>/
    ├── filtered_feature_bc_matrix.h5     (spot-level expression)
    └── spatial/
        ├── tissue_hires_image.png         (background image, ~2000 px)
        ├── tissue_lowres_image.png        (minimap image, ~600 px)
        ├── scalefactors_json.json         (coordinate scale factors)
        └── tissue_positions.csv           (spot pixel coords)

    For Visium HD (2 µm bins), Space Ranger places the above inside a
    square_002um/ subdirectory. This script checks there first:

    <sample_dir>/square_002um/
    ├── filtered_feature_bc_matrix.h5
    └── spatial/  (same structure as above)

Mode B — Pre-processed .h5ad file:

    Any AnnData where spatial coordinates live in:
      • adata.obsm['spatial']  — shape (n_obs, 2), columns [x_col, y_row]
      • adata.obs columns      — 'x'/'y' or 'pxl_col_in_fullres'/'pxl_row_in_fullres'
    Provide --image to overlay on a histology image (optional).

Usage
-----
    python spatial_viewer.py --data visium_data/Ctrl1A2
    python spatial_viewer.py --data visium_data/Ctrl1A2 --port 8888
    python spatial_viewer.py --data visium_data/Ctrl1A2 --lab-images lab_tiff_images/
    python spatial_viewer.py --data myfile.h5ad --image path/to/image.png


Install
-------
    pip install dash dash-bootstrap-components plotly scanpy pillow scipy
"""

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: Imports
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import base64
import io
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: Module-level globals (populated by main() before app.run)
# ──────────────────────────────────────────────────────────────────────────────
IMAGE: Image.Image = None          # image delivered to browser (may be downsampled)
LOWRES_IMAGE: Image.Image = None   # low-res image for minimap
IMG_WIDTH: int = 0                 # pixel width of IMAGE (browser copy)
IMG_HEIGHT: int = 0                # pixel height of IMAGE (browser copy)
COORD_WIDTH: int = 0               # coordinate-space width  (full-res, used for Plotly axes)
COORD_HEIGHT: int = 0              # coordinate-space height (full-res, used for Plotly axes)
COORDS_DF: pd.DataFrame = None     # columns: x, y (Y-flipped, hires pixel space)
GENE_MATRIX: sparse.csc_matrix = None  # (n_spots × n_genes) CSC for fast column slice
GENE_NAMES: list = []
GENE_INDEX: dict = {}              # gene_name → column index in GENE_MATRIX
SCALEFACTORS: dict = {}
UM_PER_HIRES_PX: float = None     # micrometres per hires pixel (for scale bar)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def _find_spatial_assets(base_path: Path):
    """
    Locate the spatial/ directory and filtered_feature_bc_matrix.h5 inside a
    Space Ranger output. Prefers square_002um/ (Visium HD) over the root level.

    Returns: (spatial_dir, h5_path)
    """
    candidates = [
        base_path / "square_002um",
        base_path,
    ]
    for prefix in candidates:
        spatial_dir = prefix / "spatial"
        h5_path = prefix / "filtered_feature_bc_matrix.h5"
        if spatial_dir.is_dir() and h5_path.is_file():
            return spatial_dir, h5_path

    raise FileNotFoundError(
        f"Cannot find spatial/ directory or filtered_feature_bc_matrix.h5 under {base_path}.\n"
        "Expected layout:\n"
        "  <sample>/square_002um/filtered_feature_bc_matrix.h5\n"
        "  <sample>/square_002um/spatial/tissue_hires_image.png\n"
        "  <sample>/square_002um/spatial/scalefactors_json.json\n"
        "  <sample>/square_002um/spatial/tissue_positions.csv"
    )


def _find_lab_tiff(sample_path: Path, lab_images_dir: Path) -> Path | None:
    """
    Fuzzy-match a sample directory name to a TIFF file in lab_images_dir.
    Normalises both sides to lowercase with underscores removed, so e.g.
    sample 'Ctrl1A2' matches 'Ctrl_1A2.tif'.

    Returns the matched Path, or None if no match found.
    """
    if not lab_images_dir or not lab_images_dir.is_dir():
        return None

    def normalise(s: str) -> str:
        return s.lower().replace("_", "").replace("-", "")

    sample_norm = normalise(sample_path.name)
    for tiff in lab_images_dir.glob("*.tif*"):
        if normalise(tiff.stem) == sample_norm:
            return tiff
    return None


def load_from_directory(sample_path: Path, lab_images_dir: Path = None):
    """
    Load Visium/Visium HD data from a Space Ranger output directory.

    Parameters
    ----------
    sample_path    : Path  Root sample directory (contains square_002um/ or spatial/).
    lab_images_dir : Path  Optional directory of full-res lab TIFFs. When a
                           matching TIFF is found the spot coordinates are used
                           as-is (pxl_col/row_in_fullres directly), because the
                           lab TIFFs are already in 1:1 alignment with Visium HD
                           2 µm bin coordinates.

    Returns
    -------
    image, lowres_image, coords_df, gene_matrix, gene_names, scalefactors
    """
    import scanpy as sc

    spatial_dir, h5_path = _find_spatial_assets(sample_path)

    # Scalefactors
    sf_path = spatial_dir / "scalefactors_json.json"
    with open(sf_path) as fh:
        scalefactors = json.load(fh)
    hires_scalef = scalefactors["tissue_hires_scalef"]

    # Tissue positions (prefer .csv, fall back to .parquet)
    pos_csv = spatial_dir / "tissue_positions.csv"
    pos_parq = spatial_dir / "tissue_positions.parquet"
    if pos_csv.is_file():
        pos_df = pd.read_csv(pos_csv)
    elif pos_parq.is_file():
        pos_df = pd.read_parquet(pos_parq)
    else:
        raise FileNotFoundError(f"No tissue_positions.csv or .parquet in {spatial_dir}")

    # Keep only in-tissue spots
    if "in_tissue" in pos_df.columns:
        pos_df = pos_df[pos_df["in_tissue"] == 1].copy()

    # Barcode column (may be named 'barcode' or be the first column)
    barcode_col = "barcode" if "barcode" in pos_df.columns else pos_df.columns[0]
    pos_df = pos_df.set_index(barcode_col)

    # Expression matrix
    adata = sc.read_10x_h5(str(h5_path))

    # Align barcodes
    common = pos_df.index.intersection(adata.obs_names)
    if len(common) == 0:
        raise ValueError(
            "No barcodes match between tissue_positions and filtered_feature_bc_matrix.h5. "
            "Check that you are using matching Space Ranger outputs."
        )
    pos_df = pos_df.loc[common]
    adata = adata[common]

    # ── Image and coordinate loading ────────────────────────────────────────
    lab_tiff = _find_lab_tiff(sample_path, lab_images_dir)

    if lab_tiff:
        # Lab TIFFs are in 1:1 alignment with Visium HD 2 µm bin full-res coords.
        # Use full-res pixel coords directly (no hires_scalef scaling).
        print(f"[viewer] Using lab TIFF: {lab_tiff.name}")
        Image.MAX_IMAGE_PIXELS = None   # disable decompression bomb check for large TIFFs
        full_image = Image.open(lab_tiff).convert("RGB")
        img_width, img_height = full_image.size

        x_coords = pos_df["pxl_col_in_fullres"].values
        y_coords = pos_df["pxl_row_in_fullres"].values

        # Store full-res coordinate-space dimensions BEFORE downsampling.
        # These are used for Plotly axis ranges and image placement so that
        # dots (which live in full-res pixel space) stay aligned with the image.
        scalefactors["_coord_width"]  = img_width
        scalefactors["_coord_height"] = img_height

        # Downsample TIFF to a browser-deliverable size while keeping data coords
        # in full-res space. The image is placed in data-space at its full-res
        # dimensions, and Plotly stretches the (downsampled) PNG to fill it.
        MAX_WEB_PX = 32096  # full TIFF width — JPEG compression makes this viable
        if img_width > MAX_WEB_PX:
            scale = MAX_WEB_PX / img_width
            web_w = MAX_WEB_PX
            web_h = int(img_height * scale)
            print(f"[viewer] Resampling TIFF {img_width}×{img_height} → {web_w}×{web_h} for web delivery")
            image = full_image.resize((web_w, web_h), Image.LANCZOS)
        else:
            image = full_image

        # Scale bar: microns_per_pixel from scalefactors applies to full-res coords
        scalefactors["_coord_scale"] = 1.0   # coords are already full-res

    else:
        # Fall back to bundled hires PNG
        hires_png = spatial_dir / "tissue_hires_image.png"
        if hires_png.is_file():
            image = Image.open(hires_png).convert("RGB")
        else:
            print("[viewer] Warning: no image found; using blank canvas.")
            w = int(pos_df["pxl_col_in_fullres"].max() * hires_scalef) + 50
            h = int(pos_df["pxl_row_in_fullres"].max() * hires_scalef) + 50
            image = Image.new("RGB", (w, h), (220, 210, 195))

        img_width, img_height = image.size
        x_coords = pos_df["pxl_col_in_fullres"].values * hires_scalef
        y_coords = pos_df["pxl_row_in_fullres"].values * hires_scalef
        scalefactors["_coord_scale"] = hires_scalef

    # Minimap: always a small thumbnail of whatever image we settled on
    lowres_png = spatial_dir / "tissue_lowres_image.png"
    if not lab_tiff and lowres_png.is_file():
        lowres_image = Image.open(lowres_png).convert("RGB")
    else:
        lowres_image = (full_image if lab_tiff else image).copy()
        lowres_image.thumbnail((400, 400), Image.LANCZOS)

    # Y-flip: Plotly's y-origin is bottom-left; image origin is top-left.
    # We flip y so dots rendered in data-space align with the image.
    coords_df = pd.DataFrame({
        "x": x_coords,
        "y": img_height - y_coords,
    })

    gene_matrix = sparse.csc_matrix(adata.X)
    gene_names = list(adata.var_names)

    return image, lowres_image, coords_df, gene_matrix, gene_names, scalefactors


def load_from_h5ad(h5ad_path: Path, image_path: Path = None):
    """
    Load spatial data from a pre-processed .h5ad file.

    Spatial coordinates are read from (in order of preference):
      1. adata.obsm['spatial']  — shape (n_obs, 2): [col/x, row/y]
      2. adata.obs['x'] and adata.obs['y']
      3. adata.obs['pxl_col_in_fullres'] and adata.obs['pxl_row_in_fullres']

    Returns same structure as load_from_directory.
    """
    import scanpy as sc

    adata = sc.read_h5ad(str(h5ad_path))

    # Extract coordinates
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
        x_raw = coords[:, 0].astype(float)
        y_raw = coords[:, 1].astype(float)
    elif "x" in adata.obs.columns and "y" in adata.obs.columns:
        x_raw = adata.obs["x"].values.astype(float)
        y_raw = adata.obs["y"].values.astype(float)
    elif "pxl_col_in_fullres" in adata.obs.columns:
        x_raw = adata.obs["pxl_col_in_fullres"].values.astype(float)
        y_raw = adata.obs["pxl_row_in_fullres"].values.astype(float)
    else:
        raise ValueError(
            "No spatial coordinates found in h5ad file.\n"
            "Expected one of:\n"
            "  • adata.obsm['spatial']  (shape n_obs × 2, [col, row])\n"
            "  • adata.obs columns 'x' and 'y'\n"
            "  • adata.obs columns 'pxl_col_in_fullres' and 'pxl_row_in_fullres'"
        )

    # Load image
    if image_path and image_path.is_file():
        image = Image.open(image_path).convert("RGB")
    else:
        # Blank canvas sized to coordinate extent
        w = int(x_raw.max()) + 50
        h = int(y_raw.max()) + 50
        image = Image.new("RGB", (w, h), (220, 210, 195))

    lowres_image = image.copy()
    lowres_image.thumbnail((400, 400), Image.LANCZOS)

    img_width, img_height = image.size

    coords_df = pd.DataFrame({
        "x": x_raw,
        "y": img_height - y_raw,  # Y-flip
    })

    gene_matrix = sparse.csc_matrix(adata.X)
    gene_names = list(adata.var_names)
    scalefactors = {"tissue_hires_scalef": 1.0, "microns_per_pixel": None}

    return image, lowres_image, coords_df, gene_matrix, gene_names, scalefactors


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: Figure Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _img_to_b64(img: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG data URI for Plotly."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_main_figure(layers: list, show_he: bool = True) -> go.Figure:
    """
    Build the main canvas figure from the current gene layers.

    The H&E image is placed as a layout image in data-space
    (x: 0→IMG_WIDTH, y: 0→IMG_HEIGHT). Because spots are plotted in the
    same coordinate space, they stay perfectly anchored during pan/zoom.

    uirevision='constant' tells Plotly to preserve the current viewport
    whenever this figure is replaced by a callback (e.g. when toggling a layer).
    """
    fig = go.Figure()

    if show_he:
        fig.add_layout_image(
            source=_img_to_b64(IMAGE),
            x=0,
            y=COORD_HEIGHT,  # top-left anchor in coordinate space (Y-flipped)
            xref="x",
            yref="y",
            sizex=COORD_WIDTH,   # stretch to fill full coordinate extent
            sizey=COORD_HEIGHT,
            sizing="stretch",
            layer="below",
            opacity=1.0,
        )

    # Add one WebGL scatter trace per visible gene layer
    for layer in layers:
        if layer.get("visible", True):
            fig.add_trace(_make_gene_trace(layer))

    fig.update_layout(
        xaxis=dict(
            range=[0, COORD_WIDTH],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            constrain="domain",
        ),
        yaxis=dict(
            range=[0, COORD_HEIGHT],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",   # lock aspect ratio to image
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#111111",
        paper_bgcolor=DARK,
        showlegend=False,
        uirevision="constant",   # ← critical: preserves zoom on figure update
        dragmode="pan",
    )
    return fig


def build_minimap_figure(viewport: dict = None) -> go.Figure:
    """
    Build the small minimap figure. Always shows the full tissue image.
    A yellow rectangle is drawn to indicate the current viewport in the
    main figure if viewport coords are provided.

    viewport keys: x0, x1, y0, y1  (in hires pixel / data coords)
    """
    lw, lh = LOWRES_IMAGE.size
    # Scale factor from coordinate space (full-res) to minimap pixel space
    sx = lw / COORD_WIDTH
    sy = lh / COORD_HEIGHT

    shapes = []
    if viewport:
        shapes.append(dict(
            type="rect",
            x0=np.clip(viewport["x0"] * sx, 0, lw),
            x1=np.clip(viewport["x1"] * sx, 0, lw),
            y0=np.clip(viewport["y0"] * sy, 0, lh),
            y1=np.clip(viewport["y1"] * sy, 0, lh),
            xref="x", yref="y",
            line=dict(color="yellow", width=2),
            fillcolor="rgba(255, 255, 0, 0.08)",
        ))

    fig = go.Figure()
    fig.add_layout_image(
        source=_img_to_b64(LOWRES_IMAGE),
        x=0, y=lh,
        xref="x", yref="y",
        sizex=lw, sizey=lh,
        sizing="stretch",
        layer="below",
    )
    fig.update_layout(
        xaxis=dict(range=[0, lw], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[0, lh], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", fixedrange=True),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#111111",
        paper_bgcolor=DARK,
        shapes=shapes,
        uirevision="minimap-base",
    )
    return fig


# Colorscales to cycle through as gene layers are added.
# Each is a 2-stop scale from fully transparent to fully opaque colour.
_COLORSCALES = [
    [[0, "rgba(255, 60,  60,  0)"], [1, "rgba(255, 60,  60,  1)"]],   # red
    [[0, "rgba(60,  150, 255, 0)"], [1, "rgba(60,  150, 255, 1)"]],   # blue
    [[0, "rgba(60,  220, 110, 0)"], [1, "rgba(60,  220, 110, 1)"]],   # green
    [[0, "rgba(255, 200, 50,  0)"], [1, "rgba(255, 200, 50,  1)"]],   # yellow
    [[0, "rgba(210, 80,  210, 0)"], [1, "rgba(210, 80,  210, 1)"]],   # purple
    [[0, "rgba(50,  220, 220, 0)"], [1, "rgba(50,  220, 220, 1)"]],   # cyan
    [[0, "rgba(255, 140, 0,   0)"], [1, "rgba(255, 140, 0,   1)"]],   # orange
]
_COLORSCALE_NAMES = ["Red", "Blue", "Green", "Yellow", "Purple", "Cyan", "Orange"]


def _make_gene_trace(layer: dict) -> go.Scattergl:
    """
    Build a WebGL scatter trace for one gene layer. Only spots with
    expression > 0 are plotted, coloured by normalised expression.
    Using Scattergl (WebGL) allows tens of thousands of points at 60 fps.
    """
    gene = layer["gene"]
    if gene not in GENE_INDEX:
        return go.Scattergl(x=[], y=[], mode="markers", name=gene)

    col_idx = GENE_INDEX[gene]
    # Slice one column from CSC matrix — O(nnz) operation
    expr = np.asarray(GENE_MATRIX[:, col_idx].todense()).flatten()

    mask = expr > 0
    if not mask.any():
        return go.Scattergl(x=[], y=[], mode="markers", name=gene)

    expr_nz = expr[mask]
    norm = (expr_nz - expr_nz.min()) / (expr_nz.max() - expr_nz.min() + 1e-9)

    return go.Scattergl(
        x=COORDS_DF["x"].values[mask],
        y=COORDS_DF["y"].values[mask],
        mode="markers",
        name=gene,
        marker=dict(
            color=norm,
            colorscale=layer["colorscale"],
            cmin=0,
            cmax=1,
            size=layer["radius"],
            opacity=layer["opacity"],
            showscale=False,
        ),
        hovertemplate=(
            f"<b>{gene}</b><br>"
            "count: %{customdata:.1f}<br>"
            "x: %{x:.0f} &nbsp; y: %{y:.0f}"
            "<extra></extra>"
        ),
        customdata=expr_nz,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: UI Theme Constants & Layout Builders
# ──────────────────────────────────────────────────────────────────────────────

DARK    = "#1e1e2e"
SURFACE = "#2a2a3e"
BORDER  = "#3a3a5c"
TEXT    = "#cdd6f4"
ACCENT  = "#89b4fa"


def _make_sidebar() -> html.Div:
    """Build the dark-themed left sidebar with controls."""
    return html.Div(
        style={
            "backgroundColor": DARK,
            "color": TEXT,
            "width": "270px",
            "minWidth": "270px",
            "height": "100%",
            "overflowY": "auto",
            "padding": "14px 12px",
            "borderRight": f"1px solid {BORDER}",
            "display": "flex",
            "flexDirection": "column",
            "gap": "0px",
        },
        children=[
            html.H5("Layer Manager",
                    style={"color": ACCENT, "marginBottom": "12px",
                           "fontWeight": "600", "fontSize": "15px"}),

            # ── H&E toggle ──────────────────────────────────────────────────
            html.Div(
                style={
                    "display": "flex", "alignItems": "center",
                    "justifyContent": "space-between",
                    "backgroundColor": SURFACE, "borderRadius": "6px",
                    "padding": "8px 10px", "marginBottom": "10px",
                },
                children=[
                    html.Span("H&E Histology", style={"fontSize": "13px"}),
                    dbc.Button("👁", id="he-toggle", n_clicks=0, size="sm",
                               color="secondary", outline=True,
                               style={"padding": "2px 8px", "lineHeight": "1"}),
                ],
            ),

            html.Hr(style={"borderColor": BORDER, "margin": "6px 0 10px 0"}),

            # ── Gene search & add ────────────────────────────────────────────
            html.P("Add Gene Expression Layer",
                   style={"fontSize": "11px", "color": "#888",
                          "marginBottom": "4px", "textTransform": "uppercase",
                          "letterSpacing": "0.05em"}),
            dcc.Dropdown(
                id="gene-search-dropdown",
                options=[],          # populated after data loads
                placeholder="Search gene name...",
                searchable=True,
                clearable=True,
                style={"marginBottom": "8px", "fontSize": "13px"},
            ),
            dbc.Button("+ Add Gene Layer", id="add-gene-btn", n_clicks=0,
                       color="primary", size="sm",
                       style={"width": "100%", "marginBottom": "10px",
                              "fontWeight": "600"}),

            html.Hr(style={"borderColor": BORDER, "margin": "6px 0 10px 0"}),

            # ── Gene layer list (populated by callback) ──────────────────────
            html.Div(id="layer-list"),
        ],
    )


def _make_layer_card(layer: dict, idx: int) -> html.Div:
    """Build the sidebar card for one gene layer (eye, trash, color, radius, opacity)."""
    lid = layer["id"]
    eye_label = "👁" if layer["visible"] else "🚫"
    cs_idx = layer.get("color_idx", idx % len(_COLORSCALES))

    return html.Div(
        style={
            "backgroundColor": SURFACE,
            "borderRadius": "6px",
            "padding": "10px 10px 6px 10px",
            "marginBottom": "8px",
            "border": f"1px solid {BORDER}",
        },
        children=[
            # Header: gene name | eye | trash
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "marginBottom": "8px", "gap": "6px"},
                children=[
                    html.Span(layer["gene"],
                              style={"flex": 1, "fontWeight": "700",
                                     "fontSize": "13px", "color": ACCENT,
                                     "overflow": "hidden",
                                     "textOverflow": "ellipsis",
                                     "whiteSpace": "nowrap"}),
                    dbc.Button(
                        eye_label,
                        id={"type": "eye-toggle", "index": lid},
                        n_clicks=0, size="sm", color="secondary",
                        outline=not layer["visible"],
                        style={"padding": "2px 7px", "lineHeight": "1",
                               "fontSize": "13px"},
                    ),
                    dbc.Button(
                        "🗑",
                        id={"type": "trash-btn", "index": lid},
                        n_clicks=0, size="sm", color="danger",
                        outline=True,
                        style={"padding": "2px 7px", "lineHeight": "1",
                               "fontSize": "13px"},
                    ),
                ],
            ),

            # Color scheme
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "gap": "8px", "marginBottom": "6px"},
                children=[
                    html.Small("Colour", style={"color": "#888",
                                                "minWidth": "45px"}),
                    dcc.Dropdown(
                        id={"type": "color-dropdown", "index": lid},
                        options=[{"label": n, "value": i}
                                 for i, n in enumerate(_COLORSCALE_NAMES)],
                        value=cs_idx,
                        clearable=False,
                        searchable=False,
                        style={"flex": 1, "fontSize": "12px"},
                    ),
                ],
            ),

            # Radius
            html.Div([
                html.Small(f"Radius: {layer['radius']} px",
                           id={"type": "radius-label", "index": lid},
                           style={"color": "#888"}),
                dcc.Slider(
                    id={"type": "radius-slider", "index": lid},
                    min=1, max=20, step=1, value=layer["radius"],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style={"marginBottom": "4px"}),

            # Opacity
            html.Div([
                html.Small(f"Opacity: {int(layer['opacity'] * 100)} %",
                           id={"type": "opacity-label", "index": lid},
                           style={"color": "#888"}),
                dcc.Slider(
                    id={"type": "opacity-slider", "index": lid},
                    min=0.05, max=1.0, step=0.05, value=layer["opacity"],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ]),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: Dash App + Callbacks
# ──────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,   # needed for dynamically created components
    title="Spatial Viewer",
    update_title=None,
)
server = app.server   # expose Flask server for production deployment if needed


# ── Callback 1: All gene-layer store mutations ─────────────────────────────
# One callback handles add / eye-toggle / delete / radius / opacity / colour.
# This avoids the "duplicate Output" restriction in Dash.
@app.callback(
    Output("gene-layers", "data"),
    # Triggers
    Input("add-gene-btn", "n_clicks"),
    Input({"type": "eye-toggle",    "index": ALL}, "n_clicks"),
    Input({"type": "trash-btn",     "index": ALL}, "n_clicks"),
    Input({"type": "radius-slider", "index": ALL}, "value"),
    Input({"type": "opacity-slider","index": ALL}, "value"),
    Input({"type": "color-dropdown","index": ALL}, "value"),
    # States needed to read current IDs and values
    State({"type": "radius-slider", "index": ALL}, "id"),
    State({"type": "opacity-slider","index": ALL}, "id"),
    State({"type": "color-dropdown","index": ALL}, "id"),
    State("gene-search-dropdown", "value"),
    State("gene-layers", "data"),
    prevent_initial_call=True,
)
def update_gene_layers(
    add_clicks,
    eye_clicks, trash_clicks,
    radii, opacities, color_idxs,
    radius_ids, opacity_ids, color_ids,
    gene, layers,
):
    ctx = callback_context
    if not ctx.triggered:
        return layers or []

    layers = layers or []
    triggered_prop = ctx.triggered[0]["prop_id"]   # e.g. "add-gene-btn.n_clicks"

    # ── Add new layer ──────────────────────────────────────────────────────
    if triggered_prop == "add-gene-btn.n_clicks":
        if not gene or gene not in GENE_INDEX:
            return layers
        if any(lay["gene"] == gene for lay in layers):
            return layers   # already present
        cs_idx = len(layers) % len(_COLORSCALES)
        new_layer = {
            "id":         str(uuid.uuid4())[:8],
            "gene":       gene,
            "visible":    True,
            "colorscale": _COLORSCALES[cs_idx],
            "color_idx":  cs_idx,
            "radius":     5,
            "opacity":    0.85,
        }
        return layers + [new_layer]

    # ── Parse which component fired ────────────────────────────────────────
    # triggered_prop looks like '{"index":"abc1","type":"eye-toggle"}.n_clicks'
    comp_type = None
    layer_id  = None
    try:
        prop_str, prop_name = triggered_prop.rsplit(".", 1)
        id_dict   = json.loads(prop_str)
        comp_type = id_dict.get("type")
        layer_id  = id_dict.get("index")
    except Exception:
        pass

    # Build lookup maps from layer_id → slider/dropdown current value
    radius_map  = {r["index"]: v for r, v in zip(radius_ids,  radii)     if v is not None}
    opacity_map = {o["index"]: v for o, v in zip(opacity_ids, opacities) if v is not None}
    color_map   = {c["index"]: int(v) for c, v in zip(color_ids, color_idxs) if v is not None}

    new_layers = []
    for lay in layers:
        lid = lay["id"]

        # Delete: skip this layer entirely
        if comp_type == "trash-btn" and lid == layer_id:
            continue

        updated = dict(lay)

        # Toggle visibility
        if comp_type == "eye-toggle" and lid == layer_id:
            updated["visible"] = not lay["visible"]

        # Apply slider/dropdown values (always sync current state)
        if lid in radius_map:
            updated["radius"] = radius_map[lid]
        if lid in opacity_map:
            updated["opacity"] = opacity_map[lid]
        if lid in color_map:
            ci = color_map[lid]
            updated["color_idx"]  = ci
            updated["colorscale"] = _COLORSCALES[ci % len(_COLORSCALES)]

        new_layers.append(updated)

    return new_layers


# ── Callback 2: Rebuild sidebar layer list ─────────────────────────────────
@app.callback(
    Output("layer-list", "children"),
    Input("gene-layers", "data"),
)
def rebuild_layer_list(layers):
    if not layers:
        return html.P(
            "No gene layers yet. Search for a gene above and click + Add.",
            style={"color": "#555", "fontSize": "12px", "fontStyle": "italic"},
        )
    return [_make_layer_card(lay, i) for i, lay in enumerate(layers)]


# ── Callback 3: Rebuild main figure ────────────────────────────────────────
@app.callback(
    Output("main-graph", "figure"),
    Input("gene-layers", "data"),
    Input("he-toggle", "n_clicks"),
)
def update_main_figure(layers, he_clicks):
    show_he = (he_clicks or 0) % 2 == 0   # even n_clicks = visible
    return build_main_figure(layers or [], show_he=show_he)


# ── Callback 4: Update minimap viewport rectangle ──────────────────────────
@app.callback(
    Output("minimap-graph", "figure"),
    Input("main-graph", "relayoutData"),
    prevent_initial_call=True,
)
def update_minimap(relayout_data):
    viewport = None
    if relayout_data:
        try:
            # Plotly encodes axis ranges as 'xaxis.range[0]' etc.
            x0 = relayout_data.get("xaxis.range[0]")
            x1 = relayout_data.get("xaxis.range[1]")
            y0 = relayout_data.get("yaxis.range[0]")
            y1 = relayout_data.get("yaxis.range[1]")
            if all(v is not None for v in (x0, x1, y0, y1)):
                viewport = {"x0": x0, "x1": x1, "y0": y0, "y1": y1}
        except Exception:
            pass
    return build_minimap_figure(viewport)


# ── Callback 5: Status bar ─────────────────────────────────────────────────
@app.callback(
    Output("status-bar", "children"),
    Input("main-graph", "hoverData"),
    Input("main-graph", "relayoutData"),
    prevent_initial_call=True,
)
def update_status_bar(hover_data, relayout_data):
    parts = []

    # Cursor position
    if hover_data and hover_data.get("points"):
        pt = hover_data["points"][0]
        x_px = pt.get("x", 0)
        y_px = COORD_HEIGHT - pt.get("y", 0)   # un-flip for display
        parts.append(f"X: {x_px:.0f}  Y: {y_px:.0f} px")
        if UM_PER_HIRES_PX:
            parts.append(
                f"({x_px * UM_PER_HIRES_PX:.1f}, "
                f"{y_px * UM_PER_HIRES_PX:.1f} µm)"
            )

    # Zoom level
    if relayout_data:
        x0 = relayout_data.get("xaxis.range[0]")
        x1 = relayout_data.get("xaxis.range[1]")
        if x0 is not None and x1 is not None and (x1 - x0) > 0:
            zoom = COORD_WIDTH / (x1 - x0)
            parts.append(f"Zoom: {zoom:.2f}×")

    if not parts:
        return "Hover over tissue to see coordinates"
    return "    |    ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: App Layout (set in main() after data is loaded)
# ──────────────────────────────────────────────────────────────────────────────

def _build_layout() -> html.Div:
    """
    Constructs the full Dash layout. Called after data is loaded so that
    GENE_NAMES is available for the dropdown options.
    """
    return html.Div(
        style={
            "backgroundColor": DARK,
            "height": "100vh",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
            "fontFamily": "system-ui, sans-serif",
        },
        children=[
            # Client-side stores
            dcc.Store(id="gene-layers", data=[]),

            # ── Main row: sidebar + canvas ───────────────────────────────
            html.Div(
                style={"display": "flex", "flex": 1, "overflow": "hidden",
                       "minHeight": 0},
                children=[
                    _make_sidebar(),

                    # Populate gene dropdown options now that GENE_NAMES is loaded
                    # (We inject them via dcc.Store + clientside callback, or simply
                    #  rebuild sidebar after layout is set — here we set options directly
                    #  on the Dropdown inside make_sidebar by patching after layout.)

                    html.Div(
                        dcc.Graph(
                            id="main-graph",
                            figure=build_main_figure([]),
                            style={"height": "100%", "width": "100%"},
                            config={
                                "scrollZoom": True,
                                "displayModeBar": True,
                                "modeBarButtonsToRemove": [
                                    "select2d", "lasso2d", "autoScale2d",
                                ],
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "width": 2400, "height": 2000,
                                    "filename": "spatial_viewer_export",
                                },
                            },
                        ),
                        style={"flex": 1, "height": "100%", "minWidth": 0},
                    ),
                ],
            ),

            # ── Bottom bar: minimap + status ─────────────────────────────
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "backgroundColor": SURFACE,
                    "borderTop": f"1px solid {BORDER}",
                    "padding": "4px 10px",
                    "gap": "14px",
                    "flexShrink": 0,
                },
                children=[
                    # Minimap
                    html.Div(
                        dcc.Graph(
                            id="minimap-graph",
                            figure=build_minimap_figure(),
                            style={"width": "190px", "height": "130px"},
                            config={"staticPlot": True, "displayModeBar": False},
                        ),
                        style={
                            "border": f"1px solid {BORDER}",
                            "borderRadius": "4px",
                            "overflow": "hidden",
                            "flexShrink": 0,
                        },
                    ),

                    # Status text
                    html.Div(
                        id="status-bar",
                        style={"color": TEXT, "fontSize": "12px",
                               "fontFamily": "monospace", "flex": 1},
                    ),

                    # Scale bar
                    html.Div(
                        (f"Scale: {UM_PER_HIRES_PX:.3f} µm / hires px"
                         if UM_PER_HIRES_PX else ""),
                        style={"color": "#666", "fontSize": "11px",
                               "fontFamily": "monospace", "flexShrink": 0},
                    ),
                ],
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: CLI & Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Spatial Transcriptomics Viewer — Cartoscope-like viewer for Visium HD data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data", required=True,
        help="Path to a Visium HD sample directory OR a .h5ad file",
    )
    p.add_argument(
        "--image", default=None,
        help="(h5ad mode only) Path to a histology image (PNG / JPEG / TIFF)",
    )
    p.add_argument(
        "--lab-images", default=None,
        help=(
            "Directory containing full-res lab TIFF images (e.g. lab_tiff_images/). "
            "TIFFs must be named to match the sample directory, e.g. Ctrl_1A2.tif "
            "for sample Ctrl1A2. These are expected to be in 1:1 pixel alignment "
            "with the Visium HD 2 µm bin full-res coordinates."
        ),
    )
    p.add_argument(
        "--port", type=int, default=8050,
        help="Port to serve the app on (default: 8050)",
    )
    p.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    return p.parse_args()


def main():
    global IMAGE, LOWRES_IMAGE, IMG_WIDTH, IMG_HEIGHT, COORD_WIDTH, COORD_HEIGHT
    global COORDS_DF, GENE_MATRIX, GENE_NAMES, GENE_INDEX
    global SCALEFACTORS, UM_PER_HIRES_PX

    args = _parse_args()
    data_path      = Path(args.data)
    image_path     = Path(args.image) if args.image else None
    lab_images_dir = Path(args.lab_images) if args.lab_images else None

    print(f"[viewer] Loading data from: {data_path}")
    if data_path.suffix == ".h5ad":
        (IMAGE, LOWRES_IMAGE, COORDS_DF,
         GENE_MATRIX, GENE_NAMES, SCALEFACTORS) = load_from_h5ad(
            data_path, image_path
        )
    elif data_path.is_dir():
        (IMAGE, LOWRES_IMAGE, COORDS_DF,
         GENE_MATRIX, GENE_NAMES, SCALEFACTORS) = load_from_directory(
            data_path, lab_images_dir=lab_images_dir
        )
    else:
        print(f"[viewer] ERROR: --data must be a directory or .h5ad file, got: {data_path}",
              file=sys.stderr)
        sys.exit(1)

    IMG_WIDTH, IMG_HEIGHT = IMAGE.size
    # COORD dimensions = the coordinate space Plotly axes must match.
    # When a lab TIFF was loaded, coords are full-res and the image was
    # downsampled, so these differ from IMG_WIDTH/IMG_HEIGHT.
    COORD_WIDTH  = SCALEFACTORS.get("_coord_width",  IMG_WIDTH)
    COORD_HEIGHT = SCALEFACTORS.get("_coord_height", IMG_HEIGHT)
    GENE_INDEX = {g: i for i, g in enumerate(GENE_NAMES)}

    # Compute µm per data-coordinate pixel for the status bar scale display.
    # _coord_scale == 1.0 means coords are full-res (lab TIFF path);
    # otherwise coords were scaled by tissue_hires_scalef.
    mpp          = SCALEFACTORS.get("microns_per_pixel")
    coord_scale  = SCALEFACTORS.get("_coord_scale", SCALEFACTORS.get("tissue_hires_scalef", 1.0))
    if mpp and coord_scale:
        UM_PER_HIRES_PX = mpp / coord_scale
    else:
        UM_PER_HIRES_PX = None

    print(f"[viewer] {len(COORDS_DF):,} spots  |  {len(GENE_NAMES):,} genes")
    print(f"[viewer] Image: {IMG_WIDTH} × {IMG_HEIGHT} px")
    if UM_PER_HIRES_PX:
        print(f"[viewer] Scale: {UM_PER_HIRES_PX:.4f} µm / hires px")

    # Build layout now that globals are populated (gene dropdown needs GENE_NAMES)
    app.layout = _build_layout()

    # Patch the gene dropdown options into the sidebar Dropdown.
    # Because suppress_callback_exceptions=True, we update the dropdown's
    # options via a trivial server-side callback at startup.
    @app.callback(
        Output("gene-search-dropdown", "options"),
        Input("gene-layers", "data"),   # fires once on page load (data=[])
    )
    def _init_gene_options(_):
        return [{"label": g, "value": g} for g in GENE_NAMES]

    print(f"[viewer] Starting at http://localhost:{args.port}")
    print("[viewer] Press Ctrl-C to stop.")
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
