"""Batch Analysis page.

Run statistics, curve fitting, and peak analysis across multiple loaded datasets at once.
Save and load named workflow presets.
"""

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dash_app import state
from modules.batch import batch_curve_fit, batch_peak_analysis, batch_statistics
from modules.utils.user_settings import (
    delete_preset,
    list_presets,
    load_preset,
    save_preset,
)

dash.register_page(__name__, path="/analyze-batch", title="Batch Analysis — Plottle", name="Batch Analysis")


def layout(**kwargs):
    names = state.get_dataset_names()
    ds_opts = [{"label": n, "value": n} for n in names]

    return html.Div(
        [
            html.Div(
                [
                    html.H1("Batch Analysis", className="page-title"),
                    html.P("Run analysis operations across multiple datasets simultaneously.", className="page-caption"),
                ],
                className="page-header",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label("Datasets to analyze"),
                        dcc.Dropdown(id="ba-datasets", options=ds_opts, value=names[:],
                                     multi=True, placeholder="Select datasets…"),
                    ],
                    width=8,
                ),
                className="mb-3",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(label="Statistics", tab_id="ba-tab-stats"),
                    dbc.Tab(label="Curve Fitting", tab_id="ba-tab-fit"),
                    dbc.Tab(label="Peak Analysis", tab_id="ba-tab-peaks"),
                    dbc.Tab(label="Workflow Presets", tab_id="ba-tab-presets"),
                ],
                id="ba-tabs",
                active_tab="ba-tab-stats",
            ),
            html.Div(id="ba-tab-content", className="tab-content"),
        ]
    )


@callback(Output("ba-tab-content", "children"), Input("ba-tabs", "active_tab"))
def render_tab(tab):
    if tab == "ba-tab-stats":
        return _stats_tab()
    if tab == "ba-tab-fit":
        return _fit_tab()
    if tab == "ba-tab-peaks":
        return _peaks_tab()
    if tab == "ba-tab-presets":
        return _presets_tab()
    return html.Div()


# ── Statistics tab ────────────────────────────────────────────────────────────

def _stats_tab():
    return html.Div(
        [
            html.P("Compute descriptive statistics for a shared column across all selected datasets.",
                   className="text-muted-sm mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Label("Column name (must exist in all selected datasets)"),
                         dbc.Input(id="ba-stats-col", placeholder="e.g. temperature", size="sm")],
                        width=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Button("Run Batch Statistics", id="ba-stats-btn", color="primary", className="mb-3"),
            html.Div(id="ba-stats-output"),
        ]
    )


@callback(
    Output("ba-stats-output", "children"),
    Input("ba-stats-btn", "n_clicks"),
    State("ba-datasets", "value"),
    State("ba-stats-col", "value"),
    prevent_initial_call=True,
)
def run_batch_stats(n, dataset_names, col):
    if not dataset_names:
        return dbc.Alert("Select at least one dataset.", color="warning")
    datasets = {nm: state.get_dataset(nm) for nm in dataset_names}
    datasets = {nm: d for nm, d in datasets.items() if d is not None}
    if not datasets:
        return dbc.Alert("No valid datasets found.", color="warning")
    try:
        result = batch_statistics(datasets, column=col or None)
        if isinstance(result, pd.DataFrame):
            return _df_table(result)
        return html.Pre(str(result), className="result-box")
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── Curve Fitting tab ─────────────────────────────────────────────────────────

def _fit_tab():
    return html.Div(
        [
            html.P("Fit a model to a shared X/Y column pair across all selected datasets.", className="text-muted-sm mb-3"),
            dbc.Row(
                [
                    dbc.Col([dbc.Label("X column"), dbc.Input(id="ba-fit-x", placeholder="time", size="sm")], width=3),
                    dbc.Col([dbc.Label("Y column"), dbc.Input(id="ba-fit-y", placeholder="concentration", size="sm")], width=3),
                    dbc.Col(
                        [dbc.Label("Fit type"),
                         dcc.Dropdown(id="ba-fit-type",
                                      options=[{"label": t, "value": t}
                                               for t in ["linear", "polynomial", "exponential"]],
                                      value="linear", clearable=False)],
                        width=3,
                    ),
                ],
                className="g-2 mb-3",
            ),
            dbc.Button("Run Batch Curve Fit", id="ba-fit-btn", color="primary", className="mb-3"),
            html.Div(id="ba-fit-output"),
        ]
    )


@callback(
    Output("ba-fit-output", "children"),
    Input("ba-fit-btn", "n_clicks"),
    State("ba-datasets", "value"),
    State("ba-fit-x", "value"),
    State("ba-fit-y", "value"),
    State("ba-fit-type", "value"),
    prevent_initial_call=True,
)
def run_batch_fit(n, dataset_names, xcol, ycol, fit_type):
    if not dataset_names:
        return dbc.Alert("Select at least one dataset.", color="warning")
    datasets = {nm: state.get_dataset(nm) for nm in dataset_names}
    datasets = {nm: d for nm, d in datasets.items() if d is not None}
    try:
        result = batch_curve_fit(datasets, x_column=xcol or None, y_column=ycol or None,
                                 fit_type=fit_type)
        if isinstance(result, pd.DataFrame):
            return _df_table(result)
        return html.Pre(str(result), className="result-box")
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── Peak Analysis tab ─────────────────────────────────────────────────────────

def _peaks_tab():
    return html.Div(
        [
            html.P("Detect peaks in a shared column across all selected datasets.", className="text-muted-sm mb-3"),
            dbc.Row(
                [
                    dbc.Col([dbc.Label("Column"), dbc.Input(id="ba-peaks-col", placeholder="absorbance", size="sm")], width=3),
                    dbc.Col([dbc.Label("Min height"), dbc.Input(id="ba-peaks-height", type="number", size="sm")], width=2),
                    dbc.Col([dbc.Label("Min prominence"), dbc.Input(id="ba-peaks-prom", type="number", size="sm")], width=2),
                ],
                className="g-2 mb-3",
            ),
            dbc.Button("Run Batch Peak Analysis", id="ba-peaks-btn", color="primary", className="mb-3"),
            html.Div(id="ba-peaks-output"),
        ]
    )


@callback(
    Output("ba-peaks-output", "children"),
    Input("ba-peaks-btn", "n_clicks"),
    State("ba-datasets", "value"),
    State("ba-peaks-col", "value"),
    State("ba-peaks-height", "value"),
    State("ba-peaks-prom", "value"),
    prevent_initial_call=True,
)
def run_batch_peaks(n, dataset_names, col, height, prom):
    if not dataset_names:
        return dbc.Alert("Select at least one dataset.", color="warning")
    datasets = {nm: state.get_dataset(nm) for nm in dataset_names}
    datasets = {nm: d for nm, d in datasets.items() if d is not None}
    try:
        result = batch_peak_analysis(datasets, column=col or None,
                                     height=float(height) if height else None,
                                     prominence=float(prom) if prom else None)
        if isinstance(result, pd.DataFrame):
            return _df_table(result)
        return html.Pre(str(result), className="result-box")
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── Workflow Presets tab ──────────────────────────────────────────────────────

def _presets_tab():
    try:
        presets = list_presets()
    except Exception:
        presets = []

    preset_opts = [{"label": p, "value": p} for p in presets]

    return html.Div(
        [
            html.H5("Save Current Configuration as Preset"),
            dbc.Row(
                [
                    dbc.Col([dbc.Label("Preset name"), dbc.Input(id="ba-preset-name", placeholder="my_workflow", size="sm")], width=4),
                    dbc.Col(dbc.Button("Save Preset", id="ba-preset-save-btn", color="primary", className="mt-4"), width=2),
                ],
                className="g-2 mb-3",
            ),
            html.Hr(),
            html.H5("Load / Delete Preset"),
            dbc.Row(
                [
                    dbc.Col([dbc.Label("Preset"), dcc.Dropdown(id="ba-preset-select", options=preset_opts,
                                                                placeholder="Select preset…")], width=4),
                    dbc.Col(dbc.Button("Load", id="ba-preset-load-btn", color="secondary", className="mt-4"), width=2),
                    dbc.Col(dbc.Button("Delete", id="ba-preset-del-btn", color="danger", className="mt-4"), width=2),
                ],
                className="g-2 mb-3",
            ),
            html.Div(id="ba-preset-output"),
        ]
    )


@callback(
    Output("ba-preset-output", "children"),
    Input("ba-preset-save-btn", "n_clicks"),
    Input("ba-preset-load-btn", "n_clicks"),
    Input("ba-preset-del-btn", "n_clicks"),
    State("ba-preset-name", "value"),
    State("ba-preset-select", "value"),
    State("ba-datasets", "value"),
    State("ba-stats-col", "value"),
    prevent_initial_call=True,
)
def manage_presets(save_n, load_n, del_n, name, selected, ds_names, stats_col):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered == "ba-preset-save-btn":
        if not name:
            return dbc.Alert("Enter a preset name.", color="warning")
        try:
            save_preset(name, {"datasets": ds_names or [], "stats_col": stats_col or ""})
            return dbc.Alert(f"Preset '{name}' saved.", color="success", dismissable=True)
        except Exception as e:
            return dbc.Alert(str(e), color="danger")

    if triggered == "ba-preset-load-btn":
        if not selected:
            return dbc.Alert("Select a preset.", color="warning")
        try:
            p = load_preset(selected)
            return html.Pre(str(p), className="result-box")
        except Exception as e:
            return dbc.Alert(str(e), color="danger")

    if triggered == "ba-preset-del-btn":
        if not selected:
            return dbc.Alert("Select a preset.", color="warning")
        try:
            delete_preset(selected)
            return dbc.Alert(f"Preset '{selected}' deleted.", color="success", dismissable=True)
        except Exception as e:
            return dbc.Alert(str(e), color="danger")

    return dash.no_update


# ── Helper ────────────────────────────────────────────────────────────────────

def _df_table(df: pd.DataFrame):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "var(--bg-primary)", "color": "var(--text-primary)",
                    "border": "1px solid var(--border)", "fontSize": "0.82rem"},
        style_header={"backgroundColor": "var(--bg-secondary)", "fontWeight": "600",
                      "color": "var(--text-muted)"},
        export_format="csv",
        page_size=20,
    )
