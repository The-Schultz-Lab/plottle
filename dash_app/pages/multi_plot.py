"""Multi-Plot Dashboard page.

Allows placing 1–16 plots (1×1 to 4×4 grid) simultaneously.
Each cell has independent dataset / plot-type / column configuration.
Combined PNG export of the full grid.
"""

import io
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
import base64

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dash_app import state
from modules.plotting import (
    bar_chart, box_plot, contour_plot, heatmap, histogram,
    line_plot, scatter_plot, scatter_with_regression,
    distribution_plot,
)

dash.register_page(__name__, path="/plot-multiplot", title="Multi-Plot Dashboard — Plottle", name="Multi-Plot")

_GRID_SIZES = ["1×1", "1×2", "2×1", "2×2", "2×3", "3×2", "3×3", "2×4", "4×2", "4×4"]

_SIMPLE_PLOT_TYPES = [
    "Histogram", "Line Plot", "Scatter Plot", "Bar Chart",
    "Heatmap", "Contour Plot", "Box Plot", "Scatter with Regression",
    "Distribution Plot",
]

# ── Layout ────────────────────────────────────────────────────────────────────


def layout(**kwargs):
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Multi-Plot Dashboard", className="page-title"),
                    html.P("Arrange multiple plots in a grid layout.", className="page-caption"),
                ],
                className="page-header",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Grid size"),
                            dcc.Dropdown(
                                id="mp-grid-size",
                                options=[{"label": s, "value": s} for s in _GRID_SIZES],
                                value="2×2",
                                clearable=False,
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Button("Generate All", id="mp-generate-btn",
                                   color="primary", className="mt-4 w-100"),
                        width=2,
                    ),
                    dbc.Col(
                        dbc.Button("Export PNG", id="mp-export-btn",
                                   color="secondary", className="mt-4 w-100"),
                        width=2,
                    ),
                    dbc.Col(
                        dcc.Download(id="mp-download"),
                        width=1,
                    ),
                ],
                className="g-3 mb-3",
            ),
            html.Div(id="mp-cell-configs"),
            html.Hr(),
            html.Div(id="mp-grid-output"),
        ]
    )


# ── Build cell configuration forms ───────────────────────────────────────────

@callback(Output("mp-cell-configs", "children"), Input("mp-grid-size", "value"))
def build_cell_configs(grid_size: str):
    rows, cols = map(int, grid_size.split("×"))
    n = rows * cols
    names = state.get_dataset_names()
    ds_opts = [{"label": nm, "value": nm} for nm in names]

    cards = []
    for i in range(n):
        r, c = divmod(i, cols)
        card = dbc.Card(
            dbc.CardBody(
                [
                    html.H6(f"Cell [{r+1},{c+1}]", className="text-accent mb-2"),
                    dcc.Dropdown(
                        id={"type": "mp-ds", "index": i},
                        options=ds_opts,
                        value=state._STATE["current_dataset"],
                        placeholder="Dataset…",
                        clearable=True,
                    ),
                    dcc.Dropdown(
                        id={"type": "mp-plot-type", "index": i},
                        options=[{"label": t, "value": t} for t in _SIMPLE_PLOT_TYPES],
                        value="Histogram",
                        clearable=False,
                        className="mt-1",
                    ),
                    dcc.Dropdown(
                        id={"type": "mp-xcol", "index": i},
                        options=[],
                        placeholder="X column…",
                        clearable=True,
                        className="mt-1",
                    ),
                    dcc.Dropdown(
                        id={"type": "mp-ycol", "index": i},
                        options=[],
                        placeholder="Y column…",
                        clearable=True,
                        className="mt-1",
                    ),
                    dbc.Input(
                        id={"type": "mp-title", "index": i},
                        placeholder=f"Title (optional)",
                        size="sm",
                        className="mt-1",
                    ),
                ]
            ),
            className="mb-2",
            style={"background": "var(--bg-secondary)", "border": "1px solid var(--border)"},
        )
        cards.append(dbc.Col(card, width=12 // min(cols, 4) or 3))

    # Wrap into rows of `cols` cards
    rows_out = []
    for i in range(0, n, cols):
        rows_out.append(dbc.Row(cards[i:i+cols], className="g-2"))

    return html.Div(rows_out)


# Populate column dropdowns when dataset changes
@callback(
    Output({"type": "mp-xcol", "index": dash.MATCH}, "options"),
    Output({"type": "mp-ycol", "index": dash.MATCH}, "options"),
    Input({"type": "mp-ds", "index": dash.MATCH}, "value"),
)
def populate_cols(ds_name):
    if not ds_name:
        return [], []
    data = state.get_dataset(ds_name)
    if not isinstance(data, pd.DataFrame):
        return [], []
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    all_opts = [{"label": c, "value": c} for c in data.columns]
    num_opts = [{"label": c, "value": c} for c in num_cols]
    return all_opts, num_opts


# ── Generate grid ─────────────────────────────────────────────────────────────

@callback(
    Output("mp-grid-output", "children"),
    Input("mp-generate-btn", "n_clicks"),
    State("mp-grid-size", "value"),
    State({"type": "mp-ds", "index": dash.ALL}, "value"),
    State({"type": "mp-plot-type", "index": dash.ALL}, "value"),
    State({"type": "mp-xcol", "index": dash.ALL}, "value"),
    State({"type": "mp-ycol", "index": dash.ALL}, "value"),
    State({"type": "mp-title", "index": dash.ALL}, "value"),
    prevent_initial_call=True,
)
def generate_grid(n, grid_size, ds_names, plot_types, xcols, ycols, titles):
    if not n:
        return dash.no_update
    rows, cols = map(int, grid_size.split("×"))
    n_cells = rows * cols

    try:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        plt.style.use("dark_background")
        fig.patch.set_facecolor("#1b1b1b")

        axes_flat = np.array(axes).flatten() if n_cells > 1 else [axes]

        for i, ax in enumerate(axes_flat[:n_cells]):
            ax.set_facecolor("#1b1b1b")
            ds_name = ds_names[i] if i < len(ds_names) else None
            plot_type = plot_types[i] if i < len(plot_types) else "Histogram"
            xcol = xcols[i] if i < len(xcols) else None
            ycol = ycols[i] if i < len(ycols) else None
            title = titles[i] if i < len(titles) else ""

            if not ds_name:
                ax.text(0.5, 0.5, "No dataset", ha="center", va="center",
                        color="gray", transform=ax.transAxes)
                continue

            data = state.get_dataset(ds_name)
            if data is None:
                ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                        color="gray", transform=ax.transAxes)
                continue

            try:
                _plot_to_ax(ax, data, plot_type, xcol, ycol, title)
            except Exception as exc:
                ax.text(0.5, 0.5, f"Error:\n{str(exc)[:80]}", ha="center", va="center",
                        color="#ff8888", transform=ax.transAxes, fontsize=8, wrap=True)

        # Hide unused axes
        for ax in axes_flat[n_cells:]:
            ax.set_visible(False)

        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")

        return html.Img(src=f"data:image/png;base64,{encoded}", style={"width": "100%", "borderRadius": "8px"})

    except Exception as exc:
        return dbc.Alert(f"Grid generation failed: {exc}", color="danger")


# ── Export ────────────────────────────────────────────────────────────────────

@callback(
    Output("mp-download", "data"),
    Input("mp-export-btn", "n_clicks"),
    State("mp-grid-size", "value"),
    State({"type": "mp-ds", "index": dash.ALL}, "value"),
    State({"type": "mp-plot-type", "index": dash.ALL}, "value"),
    State({"type": "mp-xcol", "index": dash.ALL}, "value"),
    State({"type": "mp-ycol", "index": dash.ALL}, "value"),
    State({"type": "mp-title", "index": dash.ALL}, "value"),
    prevent_initial_call=True,
)
def export_grid(n, grid_size, ds_names, plot_types, xcols, ycols, titles):
    if not n:
        return dash.no_update
    rows, cols = map(int, grid_size.split("×"))
    n_cells = rows * cols

    try:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        plt.style.use("dark_background")
        fig.patch.set_facecolor("#1b1b1b")
        axes_flat = np.array(axes).flatten() if n_cells > 1 else [axes]

        for i, ax in enumerate(axes_flat[:n_cells]):
            ax.set_facecolor("#1b1b1b")
            ds_name = ds_names[i] if i < len(ds_names) else None
            if not ds_name:
                continue
            data = state.get_dataset(ds_name)
            if data is None:
                continue
            try:
                _plot_to_ax(ax, data, plot_types[i], xcols[i], ycols[i], titles[i] or "")
            except Exception:
                pass

        for ax in axes_flat[n_cells:]:
            ax.set_visible(False)

        plt.tight_layout(pad=1.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return dcc.send_bytes(buf.read(), "plottle_multiplot.png")
    except Exception:
        return dash.no_update


# ── Plot to axis ──────────────────────────────────────────────────────────────

def _plot_to_ax(ax, data, plot_type: str, xcol, ycol, title: str = ""):
    """Draw a simple plot directly onto *ax*."""

    def _1d():
        if isinstance(data, pd.DataFrame):
            col = ycol or data.select_dtypes(include=[np.number]).columns[0]
            return data[col].dropna().values
        return data.flatten() if isinstance(data, np.ndarray) else np.array([])

    def _x():
        if isinstance(data, pd.DataFrame) and xcol and xcol in data.columns:
            return data[xcol].values
        return np.arange(len(_1d()))

    def _y():
        if isinstance(data, pd.DataFrame) and ycol and ycol in data.columns:
            return data[ycol].values
        return _1d()

    def _2d():
        if isinstance(data, np.ndarray) and data.ndim >= 2:
            return data
        if isinstance(data, pd.DataFrame):
            return data.select_dtypes(include=[np.number]).values
        return np.zeros((5, 5))

    if plot_type == "Histogram":
        ax.hist(_1d(), bins=20, alpha=0.8, color="#e0a3a3")
    elif plot_type == "Line Plot":
        ax.plot(_x(), _y(), linewidth=1.5)
    elif plot_type == "Scatter Plot":
        ax.scatter(_x(), _y(), alpha=0.7, s=20)
    elif plot_type == "Bar Chart":
        y = _y()
        ax.bar(np.arange(len(y)), y, alpha=0.8)
    elif plot_type == "Heatmap":
        ax.imshow(_2d(), aspect="auto", cmap="viridis")
    elif plot_type == "Contour Plot":
        Z = _2d()
        if Z.ndim == 2:
            ax.contourf(Z, cmap="viridis")
    elif plot_type == "Box Plot":
        if isinstance(data, pd.DataFrame):
            num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            cols = num_cols[:6]
            ax.boxplot([data[c].dropna().values for c in cols], labels=cols)
        else:
            ax.boxplot(_1d())
    elif plot_type == "Scatter with Regression":
        x, y = _x(), _y()
        ax.scatter(x, y, alpha=0.6, s=20)
        if len(x) >= 2:
            m, b = np.polyfit(x.astype(float), y.astype(float), 1)
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(xs, m * xs + b, "r--", linewidth=1.5)
    elif plot_type == "Distribution Plot":
        from scipy.stats import gaussian_kde
        arr = _1d()
        ax.hist(arr, bins=20, density=True, alpha=0.5, color="#e0a3a3")
        try:
            kde = gaussian_kde(arr)
            xs = np.linspace(arr.min(), arr.max(), 200)
            ax.plot(xs, kde(xs), linewidth=1.5, color="#56b4e9")
        except Exception:
            pass
    else:
        ax.plot(_x(), _y(), linewidth=1.5)

    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
