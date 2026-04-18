"""Molecular Visualization page.

Upload a Gaussian, ORCA, or Molden output file to visualize:
  • 3D molecular structure (CPK colouring, bond rendering)
  • Vibrational mode frequencies table
  • Animated normal mode displacements

Uses plottle.molecular.parsers and plottle.molecular.atom_data.
"""

import sys
from pathlib import Path
import base64
import tempfile

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dash_app import state
from plottle.molecular.parsers import parse_vibrations as parse_vibrational_output
from plottle.molecular.atom_data import atom_colors, atom_symbols, vdw_radii
# Build symbol-keyed dicts matching the page's expected interface
ELEMENT_SYMBOLS = {i: sym for i, sym in enumerate(atom_symbols)}
CPK_COLORS = {sym: atom_colors[i] for i, sym in enumerate(atom_symbols)}
CPK_RADII = {sym: vdw_radii[i] for i, sym in enumerate(atom_symbols)}

dash.register_page(__name__, path="/plot-molecular-viz", title="Molecular Viz — Plottle", name="Molecular Viz")

_SUPPORTED = ".log,.out,.molden,.fchk"


def layout(**kwargs):
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Molecular Visualization", className="page-title"),
                    html.P("Upload a Gaussian/ORCA/Molden output file to view structure and vibrational modes.", className="page-caption"),
                ],
                className="page-header",
            ),
            dcc.Upload(
                id="mv-upload",
                children=html.Div(
                    [
                        "Drag and drop or ",
                        html.A("select a Gaussian/ORCA/Molden file", style={"color": "var(--accent)"}),
                        html.Br(),
                        html.Small(f"Supported: {_SUPPORTED}", style={"color": "var(--text-muted)"}),
                    ]
                ),
                accept=_SUPPORTED,
                style={
                    "width": "100%",
                    "height": "100px",
                    "lineHeight": "100px",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "8px",
                    "borderColor": "var(--border)",
                    "textAlign": "center",
                    "background": "var(--bg-secondary)",
                    "cursor": "pointer",
                    "lineHeight": "1.6",
                    "paddingTop": "20px",
                },
            ),
            html.Div(id="mv-upload-feedback", className="mt-2"),
            html.Div(id="mv-content"),
        ]
    )


@callback(
    Output("mv-upload-feedback", "children"),
    Output("mv-content", "children"),
    Input("mv-upload", "contents"),
    State("mv-upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update

    _content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    tmp_dir = Path(tempfile.gettempdir()) / "plottle_mol"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / filename
    tmp_path.write_bytes(decoded)

    try:
        vib_data = parse_vibrational_output(str(tmp_path))
        state.set_mol_vib_data(vib_data)
        return (
            dbc.Alert(f"Parsed '{filename}' — {vib_data.program} output.", color="success", dismissable=True),
            _render_mol(vib_data),
        )
    except Exception as exc:
        return (
            dbc.Alert(f"Parse error: {exc}", color="danger"),
            html.Div(),
        )


def _render_mol(vib_data):
    """Build the structure + vibrational mode UI from a VibrationalData object."""
    # ── Structure 3D scatter ──────────────────────────────────────────────────
    geom = np.array(vib_data.geometry)  # shape (N, 3)
    atomic_nums = np.array(vib_data.atomic_numbers)  # shape (N,)

    colors = []
    sizes = []
    labels = []
    for z in atomic_nums:
        sym = ELEMENT_SYMBOLS.get(int(z), "X")
        colors.append(CPK_COLORS.get(sym, "#999999"))
        sizes.append(min(CPK_RADII.get(sym, 1.0) * 20, 30))
        labels.append(sym)

    scatter3d = go.Scatter3d(
        x=geom[:, 0], y=geom[:, 1], z=geom[:, 2],
        mode="markers+text",
        marker={"size": sizes, "color": colors, "opacity": 0.85},
        text=labels, textposition="top center",
        name="Atoms",
    )

    structure_fig = go.Figure(data=[scatter3d])
    structure_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        scene={"bgcolor": "#1b1b1b",
               "xaxis": {"title": "x (Å)"},
               "yaxis": {"title": "y (Å)"},
               "zaxis": {"title": "z (Å)"}},
        title=f"Molecular Structure — {vib_data.program}",
        height=500,
    )

    # ── Vibrational modes table ───────────────────────────────────────────────
    modes = vib_data.modes  # list of VibrationalMode
    if modes:
        mode_rows = [
            {"#": i + 1,
             "Frequency (cm⁻¹)": f"{m.frequency:.2f}",
             "IR Intensity": f"{m.ir_intensity:.4f}" if m.ir_intensity is not None else "—",
             "Symmetry": m.symmetry or "—"}
            for i, m in enumerate(modes)
        ]
        modes_table = dash_table.DataTable(
            id="mv-modes-table",
            data=mode_rows,
            columns=[{"name": c, "id": c} for c in mode_rows[0]],
            row_selectable="single",
            style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
            style_cell={"backgroundColor": "var(--bg-primary)", "color": "var(--text-primary)",
                        "border": "1px solid var(--border)", "fontSize": "0.82rem"},
            style_header={"backgroundColor": "var(--bg-secondary)", "fontWeight": "600"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "rgba(224,163,163,0.03)"}
            ],
            page_size=20,
        )
        mode_section = html.Div([
            html.H5("Vibrational Modes", className="mt-4 mb-2"),
            modes_table,
            html.P("Click a row to visualize the displacement vectors.", className="text-muted-sm mt-2"),
            html.Div(id="mv-mode-viz"),
        ])
    else:
        mode_section = dbc.Alert("No vibrational modes found in file.", color="info", className="mt-3")

    return html.Div([
        html.H5("3D Structure", className="mt-3 mb-2"),
        dcc.Graph(figure=structure_fig, id="mv-struct-graph"),
        mode_section,
    ])


@callback(
    Output("mv-mode-viz", "children"),
    Input("mv-modes-table", "selected_rows"),
    prevent_initial_call=True,
)
def show_mode(selected_rows):
    if not selected_rows:
        return html.Div()
    vib_data = state.get_mol_vib_data()
    if vib_data is None:
        return html.Div()

    idx = selected_rows[0]
    if idx >= len(vib_data.modes):
        return html.Div()
    mode = vib_data.modes[idx]
    freq = mode.frequency
    disps = np.array(mode.displacements)  # shape (N, 3)
    geom = np.array(vib_data.geometry)

    # Draw atoms + displacement arrows
    geom_end = geom + disps * 0.5
    traces = []
    colors_map = {ELEMENT_SYMBOLS.get(int(z), "X"): CPK_COLORS.get(ELEMENT_SYMBOLS.get(int(z), "X"), "#999") for z in vib_data.atomic_numbers}

    for i, (g, d, z) in enumerate(zip(geom, disps, vib_data.atomic_numbers)):
        sym = ELEMENT_SYMBOLS.get(int(z), "X")
        col = CPK_COLORS.get(sym, "#999999")
        traces.append(go.Scatter3d(
            x=[g[0], g[0] + d[0] * 0.5],
            y=[g[1], g[1] + d[1] * 0.5],
            z=[g[2], g[2] + d[2] * 0.5],
            mode="lines",
            line={"color": col, "width": 4},
            name=f"{sym}{i+1}",
            showlegend=False,
        ))

    traces.append(go.Scatter3d(
        x=geom[:, 0], y=geom[:, 1], z=geom[:, 2],
        mode="markers",
        marker={"size": 8, "color": [CPK_COLORS.get(ELEMENT_SYMBOLS.get(int(z), "X"), "#999") for z in vib_data.atomic_numbers]},
        name="Atoms",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        scene={"bgcolor": "#1b1b1b"},
        title=f"Mode {idx+1}: {freq:.2f} cm⁻¹",
        height=450,
    )
    return html.Div([
        html.H6(f"Mode {idx+1}: {freq:.2f} cm⁻¹", className="mt-3"),
        dcc.Graph(figure=fig),
    ])
