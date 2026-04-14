"""Spectroscopy page.

4 tabs mirroring the Streamlit version:
  IR / Raman   — ATR correction, spectral subtraction, cosmic ray removal
  NMR          — FFT processing, chemical shift calibration, region integration
  UV-Vis       — Beer-Lambert law, baseline subtraction, smoothing
  Mass Spec    — m/z peak detection, spectrum display, NIST WebBook lookup
"""

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dash_app import state
from modules.spectroscopy import (
    absorbance_to_transmittance,
    apply_atr_correction,
    apply_baseline_correction,
    beer_lambert,
    calibrate_ppm_axis,
    find_mz_peaks,
    integrate_nmr_regions,
    remove_cosmic_rays,
    spectral_subtraction,
    transmittance_to_absorbance,
    uv_vis_baseline_subtraction,
)
from modules.nist import fetch_ir_spectrum, get_compound_url

dash.register_page(__name__, path="/plot-spectroscopy", title="Spectroscopy — Plottle", name="Spectroscopy")


def layout(**kwargs):
    names = state.get_dataset_names()
    ds_opts = [{"label": n, "value": n} for n in names]
    current = state._STATE.get("current_dataset")

    return html.Div(
        [
            html.Div(
                [
                    html.H1("Spectroscopy", className="page-title"),
                    html.P("IR/Raman, NMR, UV-Vis, and Mass Spectrometry analysis tools.", className="page-caption"),
                ],
                className="page-header",
            ),
            dbc.Row(
                dbc.Col([dbc.Label("Dataset"), dcc.Dropdown(id="sp-dataset", options=ds_opts, value=current,
                                                             clearable=False, placeholder="Select dataset…")], width=5),
                className="mb-3",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(label="IR / Raman", tab_id="sp-ir"),
                    dbc.Tab(label="NMR", tab_id="sp-nmr"),
                    dbc.Tab(label="UV-Vis", tab_id="sp-uvvis"),
                    dbc.Tab(label="Mass Spec", tab_id="sp-ms"),
                ],
                id="sp-tabs",
                active_tab="sp-ir",
            ),
            html.Div(id="sp-tab-content", className="tab-content"),
        ]
    )


@callback(
    Output("sp-tab-content", "children"),
    Input("sp-tabs", "active_tab"),
    Input("sp-dataset", "value"),
)
def render_tab(tab, ds_name):
    df = _get_df(ds_name)
    cols = list(df.columns) if df is not None else []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist() if df is not None else []
    col_opts = [{"label": c, "value": c} for c in num_cols]
    all_ds = [{"label": n, "value": n} for n in state.get_dataset_names()]

    if tab == "sp-ir":
        return _ir_tab(col_opts, num_cols, all_ds)
    if tab == "sp-nmr":
        return _nmr_tab(col_opts, num_cols)
    if tab == "sp-uvvis":
        return _uvvis_tab(col_opts, num_cols)
    if tab == "sp-ms":
        return _ms_tab(col_opts, num_cols)
    return html.Div()


# ── IR / Raman ────────────────────────────────────────────────────────────────

def _ir_tab(col_opts, num_cols, all_ds):
    default_x = num_cols[0] if num_cols else None
    default_y = num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None)
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Label("Wavenumber column"), dcc.Dropdown(id="ir-x", options=col_opts, value=default_x, clearable=False)], width=3),
            dbc.Col([dbc.Label("Intensity column"), dcc.Dropdown(id="ir-y", options=col_opts, value=default_y, clearable=False)], width=3),
        ], className="g-2 mb-3"),
        dbc.Tabs([
            dbc.Tab(label="Display Spectrum", tab_id="ir-display"),
            dbc.Tab(label="ATR Correction", tab_id="ir-atr"),
            dbc.Tab(label="Spectral Subtraction", tab_id="ir-sub"),
            dbc.Tab(label="Cosmic Ray Removal", tab_id="ir-cosmic"),
            dbc.Tab(label="Baseline Correction", tab_id="ir-baseline"),
            dbc.Tab(label="NIST Lookup", tab_id="ir-nist"),
        ], id="ir-sub-tabs", active_tab="ir-display"),
        html.Div([
            dbc.Row([
                dbc.Col(dbc.Button("Apply", id="ir-apply-btn", color="primary"), width="auto"),
            ], className="g-2 mt-2 mb-2"),
            # ATR params
            html.Div([
                dbc.Row([
                    dbc.Col([dbc.Label("Crystal (ATR)"), dcc.Dropdown(id="ir-atr-crystal",
                              options=[{"label": c, "value": c} for c in ["ZnSe", "Ge", "Diamond", "Si"]],
                              value="ZnSe", clearable=False)], width=3),
                    dbc.Col([dbc.Label("Angle (°)"), dbc.Input(id="ir-atr-angle", type="number", value=45, size="sm")], width=2),
                ], className="g-2 mb-2", id="ir-atr-params"),
                # Subtraction params
                dbc.Row([
                    dbc.Col([dbc.Label("Background dataset"), dcc.Dropdown(id="ir-sub-bg", options=all_ds)], width=4),
                    dbc.Col([dbc.Label("BG column"), dcc.Dropdown(id="ir-sub-bgcol", options=col_opts)], width=3),
                    dbc.Col([dbc.Label("Scale factor"), dbc.Input(id="ir-sub-scale", type="number", value=1.0, step=0.05, size="sm")], width=2),
                ], className="g-2 mb-2", id="ir-sub-params"),
                # Baseline params
                dbc.Row([
                    dbc.Col([dbc.Label("Baseline method"),
                             dcc.Dropdown(id="ir-bl-method",
                                          options=[{"label": m, "value": m} for m in ["polynomial", "rubberband", "als"]],
                                          value="polynomial", clearable=False)], width=3),
                    dbc.Col([dbc.Label("Order"), dbc.Input(id="ir-bl-order", type="number", value=2, min=1, max=10, size="sm")], width=2),
                ], className="g-2 mb-2", id="ir-bl-params"),
                # NIST
                dbc.Row([
                    dbc.Col([dbc.Label("Compound name"), dbc.Input(id="ir-nist-name", placeholder="ethanol", size="sm")], width=4),
                    dbc.Col(dbc.Button("Fetch from NIST", id="ir-nist-btn", color="secondary", className="mt-4"), width=2),
                ], className="g-2 mb-2", id="ir-nist-params"),
            ]),
        ]),
        html.Div(id="ir-output"),
    ])


@callback(
    Output("ir-output", "children"),
    Input("ir-apply-btn", "n_clicks"),
    Input("ir-nist-btn", "n_clicks"),
    State("sp-dataset", "value"),
    State("ir-x", "value"),
    State("ir-y", "value"),
    State("ir-sub-tabs", "active_tab"),
    State("ir-atr-crystal", "value"),
    State("ir-atr-angle", "value"),
    State("ir-sub-bg", "value"),
    State("ir-sub-bgcol", "value"),
    State("ir-sub-scale", "value"),
    State("ir-bl-method", "value"),
    State("ir-bl-order", "value"),
    State("ir-nist-name", "value"),
    prevent_initial_call=True,
)
def ir_apply(apply_n, nist_n, ds_name, xcol, ycol, sub_tab, crystal, angle, bg_ds, bg_col, scale, bl_method, bl_order, nist_name):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered == "ir-nist-btn":
        if not nist_name:
            return dbc.Alert("Enter compound name.", color="warning")
        try:
            df = fetch_ir_spectrum(nist_name)
            if df is None or df.empty:
                return dbc.Alert("No NIST spectrum found.", color="info")
            state.add_dataset(f"NIST_{nist_name}.csv", df, metadata={"source": "NIST"})
            return dbc.Alert(f"NIST spectrum for '{nist_name}' loaded as dataset.", color="success")
        except Exception as e:
            return dbc.Alert(f"NIST error: {e}", color="danger")

    df = _get_df(ds_name)
    if df is None:
        return dbc.Alert("No DataFrame loaded.", color="warning")
    if not (xcol and ycol):
        return dbc.Alert("Select X and Y columns.", color="warning")
    x = df[xcol].values.astype(float)
    y = df[ycol].values.astype(float)

    try:
        if sub_tab == "ir-display":
            processed = y
        elif sub_tab == "ir-atr":
            processed = apply_atr_correction(x, y, crystal=crystal or "ZnSe", angle=float(angle or 45))
        elif sub_tab == "ir-sub":
            bg_df = _get_df(bg_ds)
            if bg_df is None:
                return dbc.Alert("Background dataset not found or not a DataFrame.", color="warning")
            bg_y = bg_df[bg_col].values.astype(float) if bg_col and bg_col in bg_df.columns else np.zeros_like(y)
            processed = spectral_subtraction(y, bg_y, scale=float(scale or 1.0))
        elif sub_tab == "ir-cosmic":
            processed = remove_cosmic_rays(y)
        elif sub_tab == "ir-baseline":
            processed = apply_baseline_correction(x, y, method=bl_method or "polynomial", order=int(bl_order or 2))
        else:
            processed = y
    except Exception as e:
        return dbc.Alert(str(e), color="danger")

    fig = go.Figure()
    if sub_tab != "ir-display":
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Original", opacity=0.5,
                                 line={"dash": "dash", "color": "gray"}))
    fig.add_trace(go.Scatter(x=x, y=processed, mode="lines", name="Processed",
                             line={"color": "#e0a3a3", "width": 1.5}))
    fig.update_layout(template="plotly_dark", xaxis_title="Wavenumber (cm⁻¹)",
                      yaxis_title="Intensity", xaxis={"autorange": "reversed"},
                      paper_bgcolor="rgba(0,0,0,0)")
    return dcc.Graph(figure=fig, style={"height": "420px"})


# ── NMR ───────────────────────────────────────────────────────────────────────

def _nmr_tab(col_opts, num_cols):
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Label("FID / ppm column"), dcc.Dropdown(id="nmr-x", options=col_opts,
                                                                   value=num_cols[0] if num_cols else None, clearable=False)], width=3),
            dbc.Col([dbc.Label("Signal column"), dcc.Dropdown(id="nmr-y", options=col_opts,
                                                               value=num_cols[1] if len(num_cols) > 1 else None, clearable=False)], width=3),
        ], className="g-2 mb-3"),
        dbc.Tabs([
            dbc.Tab(label="Display", tab_id="nmr-display"),
            dbc.Tab(label="FFT Processing", tab_id="nmr-fft"),
            dbc.Tab(label="ppm Calibration", tab_id="nmr-calib"),
            dbc.Tab(label="Region Integration", tab_id="nmr-int"),
        ], id="nmr-sub-tabs", active_tab="nmr-display"),
        dbc.Row([
            dbc.Col([dbc.Label("Reference peak (ppm, for calibration)"), dbc.Input(id="nmr-ref-ppm", type="number", value=0.0, size="sm")], width=3),
            dbc.Col([dbc.Label("Integration region (min,max ppm)"),
                     dbc.Row([dbc.Col(dbc.Input(id="nmr-int-min", type="number", value=0.0, size="sm"), width=6),
                              dbc.Col(dbc.Input(id="nmr-int-max", type="number", value=1.0, size="sm"), width=6)], className="g-1")], width=4),
        ], className="g-2 mb-3"),
        dbc.Button("Apply", id="nmr-apply-btn", color="primary", className="mb-3"),
        html.Div(id="nmr-output"),
    ])


@callback(Output("nmr-output", "children"),
          Input("nmr-apply-btn", "n_clicks"),
          State("sp-dataset", "value"),
          State("nmr-x", "value"), State("nmr-y", "value"),
          State("nmr-sub-tabs", "active_tab"),
          State("nmr-ref-ppm", "value"),
          State("nmr-int-min", "value"), State("nmr-int-max", "value"),
          prevent_initial_call=True)
def nmr_apply(n, ds_name, xcol, ycol, sub_tab, ref_ppm, int_min, int_max):
    df = _get_df(ds_name)
    if df is None:
        return dbc.Alert("No DataFrame loaded.", color="warning")
    x = df[xcol].values.astype(float) if xcol and xcol in df.columns else np.arange(100)
    y = df[ycol].values.astype(float) if ycol and ycol in df.columns else np.zeros(100)

    try:
        if sub_tab == "nmr-calib":
            x_cal = calibrate_ppm_axis(x, reference_ppm=float(ref_ppm or 0))
            processed = y; x_out = x_cal
        elif sub_tab == "nmr-int":
            regions = [(float(int_min or 0), float(int_max or 1))]
            result = integrate_nmr_regions(x, y, regions=regions)
            return html.Pre(str(result), className="result-box")
        elif sub_tab == "nmr-fft":
            from modules.signal import fft as sig_fft
            fft_result = sig_fft(y)
            freqs = fft_result.get("frequencies", np.arange(len(y)))
            mags = fft_result.get("magnitudes", np.zeros(len(y)))
            half = len(freqs) // 2
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs[:half], y=mags[:half], mode="lines", name="FFT"))
            fig.update_layout(template="plotly_dark", xaxis_title="Frequency", yaxis_title="Magnitude", paper_bgcolor="rgba(0,0,0,0)")
            return dcc.Graph(figure=fig, style={"height": "380px"})
        else:
            processed = y; x_out = x

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_out, y=processed, mode="lines", name="NMR", line={"color": "#e0a3a3"}))
        fig.update_layout(template="plotly_dark", xaxis_title="ppm", yaxis_title="Intensity",
                          xaxis={"autorange": "reversed"}, paper_bgcolor="rgba(0,0,0,0)")
        return dcc.Graph(figure=fig, style={"height": "380px"})
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── UV-Vis ────────────────────────────────────────────────────────────────────

def _uvvis_tab(col_opts, num_cols):
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Label("Wavelength column"), dcc.Dropdown(id="uv-x", options=col_opts,
                                                                   value=num_cols[0] if num_cols else None, clearable=False)], width=3),
            dbc.Col([dbc.Label("Absorbance column"), dcc.Dropdown(id="uv-y", options=col_opts,
                                                                   value=num_cols[1] if len(num_cols) > 1 else None, clearable=False)], width=3),
        ], className="g-2 mb-3"),
        dbc.Row([
            dbc.Col([dbc.Label("Molar absorptivity ε (L/mol·cm)"), dbc.Input(id="uv-eps", type="number", value=1000, size="sm")], width=3),
            dbc.Col([dbc.Label("Path length (cm)"), dbc.Input(id="uv-path", type="number", value=1.0, step=0.1, size="sm")], width=2),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col(dbc.Button("Display Spectrum", id="uv-display-btn", color="secondary"), width="auto"),
            dbc.Col(dbc.Button("Beer-Lambert Concentration", id="uv-bl-btn", color="primary"), width="auto"),
            dbc.Col(dbc.Button("Baseline Subtraction", id="uv-base-btn", color="secondary"), width="auto"),
        ], className="g-2 mb-3"),
        html.Div(id="uv-output"),
    ])


@callback(Output("uv-output", "children"),
          Input("uv-display-btn", "n_clicks"),
          Input("uv-bl-btn", "n_clicks"),
          Input("uv-base-btn", "n_clicks"),
          State("sp-dataset", "value"),
          State("uv-x", "value"), State("uv-y", "value"),
          State("uv-eps", "value"), State("uv-path", "value"),
          prevent_initial_call=True)
def uv_apply(disp_n, bl_n, base_n, ds_name, xcol, ycol, eps, path):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    df = _get_df(ds_name)
    if df is None:
        return dbc.Alert("No DataFrame loaded.", color="warning")
    x = df[xcol].values.astype(float) if xcol and xcol in df.columns else np.arange(100)
    y = df[ycol].values.astype(float) if ycol and ycol in df.columns else np.zeros(100)

    try:
        if triggered == "uv-bl-btn":
            result = beer_lambert(y, epsilon=float(eps or 1000), path_length=float(path or 1.0))
            return html.Pre(str(result), className="result-box")
        if triggered == "uv-base-btn":
            y = uv_vis_baseline_subtraction(x, y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="UV-Vis", line={"color": "#e0a3a3"}))
        fig.update_layout(template="plotly_dark", xaxis_title="Wavelength (nm)",
                          yaxis_title="Absorbance", paper_bgcolor="rgba(0,0,0,0)")
        return dcc.Graph(figure=fig, style={"height": "380px"})
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── Mass Spec ─────────────────────────────────────────────────────────────────

def _ms_tab(col_opts, num_cols):
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Label("m/z column"), dcc.Dropdown(id="ms-x", options=col_opts,
                                                             value=num_cols[0] if num_cols else None, clearable=False)], width=3),
            dbc.Col([dbc.Label("Intensity column"), dcc.Dropdown(id="ms-y", options=col_opts,
                                                                  value=num_cols[1] if len(num_cols) > 1 else None, clearable=False)], width=3),
            dbc.Col([dbc.Label("Min peak height"), dbc.Input(id="ms-height", type="number", value=0.05, step=0.01, size="sm")], width=2),
        ], className="g-2 mb-3"),
        dbc.Row([
            dbc.Col(dbc.Button("Display Spectrum", id="ms-display-btn", color="primary"), width="auto"),
            dbc.Col(dbc.Button("Find Peaks", id="ms-peaks-btn", color="secondary"), width="auto"),
        ], className="g-2 mb-3"),
        html.Div(id="ms-output"),
    ])


@callback(Output("ms-output", "children"),
          Input("ms-display-btn", "n_clicks"),
          Input("ms-peaks-btn", "n_clicks"),
          State("sp-dataset", "value"),
          State("ms-x", "value"), State("ms-y", "value"),
          State("ms-height", "value"),
          prevent_initial_call=True)
def ms_apply(disp_n, peaks_n, ds_name, xcol, ycol, height):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    df = _get_df(ds_name)
    if df is None:
        return dbc.Alert("No DataFrame loaded.", color="warning")
    x = df[xcol].values.astype(float) if xcol and xcol in df.columns else np.arange(100)
    y = df[ycol].values.astype(float) if ycol and ycol in df.columns else np.zeros(100)

    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=y, name="m/z", marker_color="#e0a3a3"))

        if triggered == "ms-peaks-btn":
            peaks = find_mz_peaks(x, y, min_height=float(height or 0.05))
            if peaks and "peaks" in peaks:
                peak_x = peaks["peaks"]
                peak_y = y[[np.argmin(np.abs(x - px)) for px in peak_x]]
                fig.add_trace(go.Scatter(x=peak_x, y=peak_y, mode="markers+text",
                                         text=[f"{px:.2f}" for px in peak_x],
                                         textposition="top center",
                                         name="Peaks", marker={"size": 8, "color": "#56b4e9"}))

        fig.update_layout(template="plotly_dark", xaxis_title="m/z",
                          yaxis_title="Intensity", paper_bgcolor="rgba(0,0,0,0)")
        return dcc.Graph(figure=fig, style={"height": "380px"})
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_df(ds_name):
    if not ds_name:
        return None
    data = state.get_dataset(ds_name)
    return data if isinstance(data, pd.DataFrame) else None
