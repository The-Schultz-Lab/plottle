"""Analysis Tools page.

8 tabs mirroring the Streamlit version:
  Statistics         — descriptive stats, normality test
  Distribution       — fit distribution to data
  Curve Fitting      — linear, polynomial, exponential, custom
  Optimization       — minimize function, find roots
  Linear Algebra     — eigenvalues, linear system, decomposition
  Signal Processing  — smoothing, filtering, FFT, derivatives, baseline
  Peak Analysis      — find peaks, FWHM, integrate, multipeak fit
  Statistical Tests  — t-tests, ANOVA, correlations, non-parametric
"""

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dash_app import state
from plottle.math import (
    anova_oneway,
    bonferroni_correction,
    calculate_statistics,
    check_normality,
    chi_square_independence,
    compute_eigenvalues,
    find_roots,
    fit_custom,
    fit_distribution,
    fit_exponential,
    fit_linear,
    fit_polynomial,
    kruskal_wallis,
    mannwhitney_u,
    matrix_decomposition,
    minimize_function,
    pearson_correlation,
    solve_linear_system,
    spearman_correlation,
    ttest_one_sample,
    ttest_paired,
    ttest_two_sample,
    tukey_hsd,
    wilcoxon_signed_rank,
)
from plottle.peaks import compute_fwhm, find_peaks as pk_find_peaks, fit_multipeak, integrate_peaks
from plottle.signal import (
    baseline_als,
    baseline_polynomial,
    baseline_rolling_ball,
    derivative as sig_derivative,
    fft as sig_fft,
    filter_bandpass,
    filter_bandstop,
    filter_highpass,
    filter_lowpass,
    interpolate as sig_interpolate,
    smooth_gaussian,
    smooth_moving_average,
    smooth_savitzky_golay,
)

dash.register_page(__name__, path="/analyze-single", title="Analysis Tools — Plottle", name="Analysis Tools")


def layout(**kwargs):
    names = state.get_dataset_names()
    ds_opts = [{"label": n, "value": n} for n in names]
    current = state._STATE.get("current_dataset")

    return html.Div(
        [
            html.Div(
                [
                    html.H1("Analysis Tools", className="page-title"),
                    html.P("Statistical analysis, curve fitting, signal processing, and more.", className="page-caption"),
                ],
                className="page-header",
            ),
            dbc.Row(
                dbc.Col([
                    dbc.Label("Dataset"),
                    dcc.Dropdown(id="at-dataset", options=ds_opts, value=current,
                                 clearable=False, placeholder="Select dataset…"),
                ], width=5),
                className="mb-3",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(label="Statistics", tab_id="tab-stats"),
                    dbc.Tab(label="Distribution", tab_id="tab-dist"),
                    dbc.Tab(label="Curve Fitting", tab_id="tab-fit"),
                    dbc.Tab(label="Optimization", tab_id="tab-optim"),
                    dbc.Tab(label="Linear Algebra", tab_id="tab-linalg"),
                    dbc.Tab(label="Signal Processing", tab_id="tab-signal"),
                    dbc.Tab(label="Peak Analysis", tab_id="tab-peaks"),
                    dbc.Tab(label="Statistical Tests", tab_id="tab-tests"),
                ],
                id="at-tabs",
                active_tab="tab-stats",
            ),
            html.Div(id="at-tab-content", className="tab-content"),
        ]
    )


@callback(
    Output("at-tab-content", "children"),
    Input("at-tabs", "active_tab"),
    Input("at-dataset", "value"),
)
def render_tab(tab, ds_name):
    data = state.get_dataset(ds_name) if ds_name else None
    num_cols = _num_cols(data)
    col_opts = [{"label": c, "value": c} for c in num_cols]

    if tab == "tab-stats":
        return _stats_tab(col_opts, num_cols)
    if tab == "tab-dist":
        return _dist_tab(col_opts, num_cols)
    if tab == "tab-fit":
        return _fit_tab(col_opts, num_cols)
    if tab == "tab-optim":
        return _optim_tab()
    if tab == "tab-linalg":
        return _linalg_tab(col_opts, num_cols, data)
    if tab == "tab-signal":
        return _signal_tab(col_opts, num_cols)
    if tab == "tab-peaks":
        return _peaks_tab(col_opts, num_cols)
    if tab == "tab-tests":
        return _tests_tab(col_opts, num_cols)
    return html.Div()


# ── Statistics tab ────────────────────────────────────────────────────────────

def _stats_tab(col_opts, num_cols):
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Column"),
                dcc.Dropdown(id="at-stats-col", options=col_opts,
                             value=num_cols[0] if num_cols else None, clearable=False),
            ], width=4),
            dbc.Col([
                dbc.Label("Action"),
                dbc.ButtonGroup([
                    dbc.Button("Descriptive Stats", id="at-stats-calc-btn", color="primary"),
                    dbc.Button("Test Normality", id="at-stats-norm-btn", color="secondary"),
                ]),
            ], width=6),
        ], className="g-2 mb-3"),
        html.Div(id="at-stats-output"),
    ])


@callback(Output("at-stats-output", "children"),
          Input("at-stats-calc-btn", "n_clicks"),
          Input("at-stats-norm-btn", "n_clicks"),
          State("at-dataset", "value"),
          State("at-stats-col", "value"),
          prevent_initial_call=True)
def compute_stats(calc_n, norm_n, ds_name, col):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    arr = _get_arr(ds_name, col)
    if arr is None:
        return dbc.Alert("No data.", color="warning")

    if triggered == "at-stats-calc-btn":
        try:
            stats = calculate_statistics(arr)
            state.add_analysis_result({"type": "statistics", "dataset": ds_name, "column": col, "results": stats})
            return _stats_cards(stats)
        except Exception as e:
            return dbc.Alert(str(e), color="danger")

    if triggered == "at-stats-norm-btn":
        try:
            result = check_normality(arr)
            color = "success" if result["is_normal"] else "warning"
            return dbc.Alert([
                html.Strong("Shapiro-Wilk Normality Test"), html.Br(),
                f"Statistic: {result['statistic']:.6f}", html.Br(),
                f"P-value: {result['p_value']:.6f}", html.Br(),
                "Result: " + ("Normally distributed (p > 0.05)" if result["is_normal"] else "Not normally distributed (p ≤ 0.05)"),
            ], color=color)
        except Exception as e:
            return dbc.Alert(str(e), color="danger")
    return dash.no_update


def _stats_cards(stats: dict):
    keys = [("mean", "Mean"), ("median", "Median"), ("std", "Std Dev"),
            ("min", "Min"), ("max", "Max"), ("range", "Range"),
            ("q1", "Q1 (25%)"), ("q3", "Q3 (75%)"), ("iqr", "IQR")]
    cols = [
        dbc.Col(
            html.Div([html.Div(f"{stats.get(k, '—'):.4g}", className="metric-value"),
                      html.Div(label, className="metric-label")], className="metric-card"),
            width=4,
        )
        for k, label in keys if k in stats
    ]
    rows = []
    for i in range(0, len(cols), 3):
        rows.append(dbc.Row(cols[i:i+3], className="g-2 mb-2"))
    return html.Div(rows)


# ── Distribution tab ──────────────────────────────────────────────────────────

def _dist_tab(col_opts, num_cols):
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Label("Column"), dcc.Dropdown(id="at-dist-col", options=col_opts,
                                                        value=num_cols[0] if num_cols else None, clearable=False)], width=3),
            dbc.Col([dbc.Label("Distribution"),
                     dcc.Dropdown(id="at-dist-name",
                                  options=[{"label": d.capitalize(), "value": d}
                                           for d in ["norm", "expon", "gamma", "lognorm", "beta"]],
                                  value="norm", clearable=False)], width=3),
        ], className="g-2 mb-3"),
        dbc.Button("Fit Distribution", id="at-dist-fit-btn", color="primary", className="mb-3"),
        html.Div(id="at-dist-output"),
    ])


@callback(Output("at-dist-output", "children"),
          Input("at-dist-fit-btn", "n_clicks"),
          State("at-dataset", "value"),
          State("at-dist-col", "value"),
          State("at-dist-name", "value"),
          prevent_initial_call=True)
def fit_dist(n, ds_name, col, dist_name):
    arr = _get_arr(ds_name, col)
    if arr is None:
        return dbc.Alert("No data.", color="warning")
    try:
        result = fit_distribution(arr, distribution=dist_name)
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in result.get("params", {}).items())
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=arr, nbinsx=30, histnorm="probability density",
                                   name="Data", opacity=0.6))
        xs = np.linspace(arr.min(), arr.max(), 300)
        if "pdf_values" in result:
            fig.add_trace(go.Scatter(x=xs, y=result["pdf_values"], name=f"Fitted {dist_name}", mode="lines",
                                     line={"color": "#e0a3a3", "width": 2}))
        fig.update_layout(template="plotly_dark", title=f"Fitted {dist_name} distribution", paper_bgcolor="rgba(0,0,0,0)")
        return html.Div([
            dbc.Alert([html.Strong(f"Fitted {dist_name}: "), params_str,
                       html.Br(), f"AIC: {result.get('aic', '—'):.4f}" if "aic" in result else ""], color="info"),
            dcc.Graph(figure=fig, style={"height": "400px"}),
        ])
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── Curve Fitting tab ─────────────────────────────────────────────────────────

def _fit_tab(col_opts, num_cols):
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Label("X column"), dcc.Dropdown(id="at-fit-x", options=col_opts,
                                                          value=num_cols[0] if num_cols else None, clearable=False)], width=3),
            dbc.Col([dbc.Label("Y column"), dcc.Dropdown(id="at-fit-y", options=col_opts,
                                                          value=num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None), clearable=False)], width=3),
            dbc.Col([dbc.Label("Fit type"),
                     dcc.Dropdown(id="at-fit-type",
                                  options=[{"label": t, "value": t}
                                           for t in ["Linear", "Polynomial", "Exponential", "Custom"]],
                                  value="Linear", clearable=False)], width=3),
            dbc.Col([dbc.Label("Poly degree (if polynomial)"),
                     dbc.Input(id="at-fit-degree", type="number", value=2, min=1, max=10, size="sm")], width=3),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col([dbc.Label("Custom expression (use x as variable)"),
                     dbc.Input(id="at-fit-expr", placeholder="a*np.exp(-b*x) + c", type="text", size="sm")], width=6),
        ], className="g-2 mb-3"),
        dbc.Button("Fit", id="at-fit-btn", color="primary", className="mb-3"),
        html.Div(id="at-fit-output"),
    ])


@callback(Output("at-fit-output", "children"),
          Input("at-fit-btn", "n_clicks"),
          State("at-dataset", "value"),
          State("at-fit-x", "value"),
          State("at-fit-y", "value"),
          State("at-fit-type", "value"),
          State("at-fit-degree", "value"),
          State("at-fit-expr", "value"),
          prevent_initial_call=True)
def do_fit(n, ds_name, xcol, ycol, fit_type, degree, expr):
    x = _get_arr(ds_name, xcol)
    y = _get_arr(ds_name, ycol)
    if x is None or y is None:
        return dbc.Alert("Need X and Y columns.", color="warning")
    try:
        if fit_type == "Linear":
            result = fit_linear(x, y)
        elif fit_type == "Polynomial":
            result = fit_polynomial(x, y, degree=int(degree or 2))
        elif fit_type == "Exponential":
            result = fit_exponential(x, y)
        elif fit_type == "Custom" and expr:
            result = fit_custom(x, y, expression=expr)
        else:
            return dbc.Alert("Provide an expression for Custom fit.", color="warning")

        state.add_analysis_result({"type": "curve_fit", "fit_type": fit_type, "dataset": ds_name, "results": result})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data", marker={"size": 5, "opacity": 0.7}))
        if "y_fit" in result:
            fig.add_trace(go.Scatter(x=x, y=result["y_fit"], mode="lines", name="Fit",
                                     line={"color": "#e0a3a3", "width": 2}))
        fig.update_layout(template="plotly_dark", title=f"{fit_type} Fit", paper_bgcolor="rgba(0,0,0,0)")

        info = {k: v for k, v in result.items() if k not in ("y_fit",)}
        return html.Div([
            html.Pre(str(info), className="result-box mb-2"),
            dcc.Graph(figure=fig, style={"height": "380px"}),
        ])
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── Optimization tab ──────────────────────────────────────────────────────────

def _optim_tab():
    return html.Div([
        html.H5("Minimize Function"),
        dbc.Row([
            dbc.Col([dbc.Label("f(x) expression"), dbc.Input(id="at-optim-expr", placeholder="(x-2)**2 + 3",
                                                               type="text", size="sm")], width=5),
            dbc.Col([dbc.Label("Initial guess"), dbc.Input(id="at-optim-x0", type="number", value=0, size="sm")], width=2),
            dbc.Col([dbc.Label("Algorithm"),
                     dcc.Dropdown(id="at-optim-alg",
                                  options=[{"label": m, "value": m} for m in ["Nelder-Mead", "BFGS", "Powell"]],
                                  value="Nelder-Mead", clearable=False)], width=3),
        ], className="g-2 mb-2"),
        dbc.Button("Minimize", id="at-optim-min-btn", color="primary", className="mb-3"),
        html.Hr(),
        html.H5("Find Roots"),
        dbc.Row([
            dbc.Col([dbc.Label("f(x) expression"), dbc.Input(id="at-roots-expr", placeholder="x**3 - x - 2",
                                                               type="text", size="sm")], width=5),
            dbc.Col([dbc.Label("x range [min, max]"),
                     dbc.Row([dbc.Col(dbc.Input(id="at-roots-xmin", type="number", value=-5, size="sm"), width=6),
                              dbc.Col(dbc.Input(id="at-roots-xmax", type="number", value=5, size="sm"), width=6)], className="g-1")], width=4),
        ], className="g-2 mb-2"),
        dbc.Button("Find Roots", id="at-roots-btn", color="secondary", className="mb-3"),
        html.Div(id="at-optim-output"),
    ])


@callback(Output("at-optim-output", "children"),
          Input("at-optim-min-btn", "n_clicks"),
          Input("at-roots-btn", "n_clicks"),
          State("at-optim-expr", "value"),
          State("at-optim-x0", "value"),
          State("at-optim-alg", "value"),
          State("at-roots-expr", "value"),
          State("at-roots-xmin", "value"),
          State("at-roots-xmax", "value"),
          prevent_initial_call=True)
def do_optim(min_n, roots_n, min_expr, x0, alg, roots_expr, xmin, xmax):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered == "at-optim-min-btn" and min_expr:
        try:
            result = minimize_function(min_expr, x0=float(x0 or 0), method=alg)
            return html.Pre(str(result), className="result-box")
        except Exception as e:
            return dbc.Alert(str(e), color="danger")

    if triggered == "at-roots-btn" and roots_expr:
        try:
            result = find_roots(roots_expr, x_range=(float(xmin or -5), float(xmax or 5)))
            return html.Pre(str(result), className="result-box")
        except Exception as e:
            return dbc.Alert(str(e), color="danger")
    return dash.no_update


# ── Linear Algebra tab ────────────────────────────────────────────────────────

def _linalg_tab(col_opts, num_cols, data):
    return html.Div([
        dbc.Tabs([
            dbc.Tab(label="Eigenvalues", tab_id="la-eig"),
            dbc.Tab(label="Solve Ax=b", tab_id="la-solve"),
            dbc.Tab(label="Decomposition", tab_id="la-decomp"),
        ], id="la-tabs", active_tab="la-eig"),
        html.Div(id="la-content", className="mt-3"),
    ])


@callback(Output("la-content", "children"),
          Input("la-tabs", "active_tab"),
          State("at-dataset", "value"))
def la_tab(tab, ds_name):
    data = state.get_dataset(ds_name) if ds_name else None
    num_cols = _num_cols(data)
    col_opts = [{"label": c, "value": c} for c in num_cols]

    if tab == "la-eig":
        return html.Div([
            dbc.Label("Use numeric columns as matrix rows"),
            dbc.Button("Compute Eigenvalues", id="la-eig-btn", color="primary", className="ms-3"),
            html.Div(id="la-eig-output", className="mt-2"),
        ])
    if tab == "la-solve":
        return html.Div([
            html.P("Enter matrix A and vector b as space-separated rows.", className="text-muted-sm"),
            dbc.Row([
                dbc.Col([dbc.Label("Matrix A (one row per line)"),
                         dbc.Textarea(id="la-A", placeholder="1 2 3\n4 5 6\n7 8 9", rows=4)], width=5),
                dbc.Col([dbc.Label("Vector b (one value per line)"),
                         dbc.Textarea(id="la-b", placeholder="1\n2\n3", rows=4)], width=3),
            ], className="g-2 mb-2"),
            dbc.Button("Solve", id="la-solve-btn", color="primary", className="mb-2"),
            html.Div(id="la-solve-output"),
        ])
    if tab == "la-decomp":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Decomposition"),
                         dcc.Dropdown(id="la-decomp-type",
                                      options=[{"label": d, "value": d} for d in ["SVD", "QR", "LU", "Cholesky"]],
                                      value="SVD", clearable=False)], width=3),
            ], className="g-2 mb-2"),
            dbc.Button("Decompose", id="la-decomp-btn", color="primary", className="mb-2"),
            html.Div(id="la-decomp-output"),
        ])
    return html.Div()


@callback(Output("la-eig-output", "children"),
          Input("la-eig-btn", "n_clicks"),
          State("at-dataset", "value"), prevent_initial_call=True)
def eig_compute(n, ds_name):
    data = state.get_dataset(ds_name)
    num_cols = _num_cols(data)
    if not num_cols:
        return dbc.Alert("No numeric columns.", color="warning")
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        return dbc.Alert("Need DataFrame or array.", color="warning")
    mat = data[num_cols].values if isinstance(data, pd.DataFrame) else data
    if mat.ndim == 1:
        mat = np.diag(mat)
    try:
        result = compute_eigenvalues(mat)
        return html.Pre(str(result), className="result-box")
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


@callback(Output("la-solve-output", "children"),
          Input("la-solve-btn", "n_clicks"),
          State("la-A", "value"), State("la-b", "value"), prevent_initial_call=True)
def la_solve(n, A_str, b_str):
    if not (A_str and b_str):
        return dbc.Alert("Enter A and b.", color="warning")
    try:
        A = np.array([[float(v) for v in row.split()] for row in A_str.strip().split("\n")])
        b = np.array([float(v) for v in b_str.strip().split()])
        result = solve_linear_system(A, b)
        return html.Pre(str(result), className="result-box")
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


@callback(Output("la-decomp-output", "children"),
          Input("la-decomp-btn", "n_clicks"),
          State("at-dataset", "value"),
          State("la-decomp-type", "value"), prevent_initial_call=True)
def la_decomp(n, ds_name, dtype):
    data = state.get_dataset(ds_name)
    num_cols = _num_cols(data)
    if not num_cols:
        return dbc.Alert("No numeric data.", color="warning")
    mat = data[num_cols].values if isinstance(data, pd.DataFrame) else data
    if isinstance(data, np.ndarray) and data.ndim == 1:
        mat = np.diag(data)
    try:
        result = matrix_decomposition(mat, method=dtype)
        lines = [f"{k}: shape {v.shape}" for k, v in result.items() if isinstance(v, np.ndarray)]
        return html.Pre("\n".join(lines) + "\n\n" + str({k: v.tolist() for k, v in result.items() if isinstance(v, np.ndarray) and v.size < 50}), className="result-box")
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── Signal Processing tab ─────────────────────────────────────────────────────

def _signal_tab(col_opts, num_cols):
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Label("Column"), dcc.Dropdown(id="at-sig-col", options=col_opts,
                                                        value=num_cols[0] if num_cols else None, clearable=False)], width=3),
            dbc.Col([dbc.Label("Operation"),
                     dcc.Dropdown(id="at-sig-op",
                                  options=[{"label": o, "value": o} for o in [
                                      "Smooth (Moving Average)", "Smooth (Savitzky-Golay)", "Smooth (Gaussian)",
                                      "Filter (Lowpass)", "Filter (Highpass)", "Filter (Bandpass)", "Filter (Bandstop)",
                                      "FFT", "Derivative", "Baseline (Polynomial)", "Baseline (Rolling Ball)", "Baseline (ALS)",
                                      "Interpolate",
                                  ]],
                                  value="Smooth (Moving Average)", clearable=False)], width=5),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col([dbc.Label("Window / Order"),
                     dbc.Input(id="at-sig-window", type="number", value=11, min=3, max=101, step=2, size="sm")], width=3),
            dbc.Col([dbc.Label("Cutoff freq (0–1, for filters)"),
                     dbc.Input(id="at-sig-cutoff", type="number", value=0.1, min=0.001, max=0.499, step=0.01, size="sm")], width=3),
        ], className="g-2 mb-3"),
        dbc.Button("Apply", id="at-sig-apply-btn", color="primary", className="mb-3"),
        html.Div(id="at-sig-output"),
    ])


@callback(Output("at-sig-output", "children"),
          Input("at-sig-apply-btn", "n_clicks"),
          State("at-dataset", "value"),
          State("at-sig-col", "value"),
          State("at-sig-op", "value"),
          State("at-sig-window", "value"),
          State("at-sig-cutoff", "value"),
          prevent_initial_call=True)
def apply_signal(n, ds_name, col, op, window, cutoff):
    arr = _get_arr(ds_name, col)
    if arr is None:
        return dbc.Alert("No data.", color="warning")
    win = int(window or 11)
    fc = float(cutoff or 0.1)

    try:
        if op == "Smooth (Moving Average)":
            result = smooth_moving_average(arr, window_size=win)
        elif op == "Smooth (Savitzky-Golay)":
            result = smooth_savitzky_golay(arr, window_length=win)
        elif op == "Smooth (Gaussian)":
            result = smooth_gaussian(arr, sigma=win / 4)
        elif op == "Filter (Lowpass)":
            result = filter_lowpass(arr, cutoff=fc)
        elif op == "Filter (Highpass)":
            result = filter_highpass(arr, cutoff=fc)
        elif op == "Filter (Bandpass)":
            result = filter_bandpass(arr, low=fc, high=min(fc * 5, 0.45))
        elif op == "Filter (Bandstop)":
            result = filter_bandstop(arr, low=fc, high=min(fc * 5, 0.45))
        elif op == "FFT":
            result = sig_fft(arr)
            freq = result.get("frequencies", np.arange(len(arr)))
            mag = result.get("magnitudes", np.zeros(len(arr)))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freq[:len(freq)//2], y=mag[:len(mag)//2], mode="lines", name="FFT magnitude"))
            fig.update_layout(template="plotly_dark", title="FFT", xaxis_title="Frequency", yaxis_title="Magnitude", paper_bgcolor="rgba(0,0,0,0)")
            return dcc.Graph(figure=fig, style={"height": "380px"})
        elif op == "Derivative":
            result = sig_derivative(arr)
        elif op == "Baseline (Polynomial)":
            result = baseline_polynomial(arr, degree=2)
        elif op == "Baseline (Rolling Ball)":
            result = baseline_rolling_ball(arr, radius=win)
        elif op == "Baseline (ALS)":
            result = baseline_als(arr)
        elif op == "Interpolate":
            result = sig_interpolate(arr, new_length=len(arr) * 2)
        else:
            return dbc.Alert("Unknown operation.", color="warning")

        if isinstance(result, np.ndarray):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=arr, mode="lines", name="Original", opacity=0.5))
            fig.add_trace(go.Scatter(y=result, mode="lines", name=op, line={"color": "#e0a3a3"}))
            fig.update_layout(template="plotly_dark", title=op, paper_bgcolor="rgba(0,0,0,0)")
            return dcc.Graph(figure=fig, style={"height": "380px"})
        return html.Pre(str(result), className="result-box")

    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── Peak Analysis tab ─────────────────────────────────────────────────────────

def _peaks_tab(col_opts, num_cols):
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Label("Column"), dcc.Dropdown(id="at-peaks-col", options=col_opts,
                                                        value=num_cols[0] if num_cols else None, clearable=False)], width=3),
            dbc.Col([dbc.Label("Min height"), dbc.Input(id="at-peaks-height", type="number", size="sm")], width=2),
            dbc.Col([dbc.Label("Min prominence"), dbc.Input(id="at-peaks-prom", type="number", size="sm")], width=2),
            dbc.Col([dbc.Label("Min distance"), dbc.Input(id="at-peaks-dist", type="number", value=5, min=1, size="sm")], width=2),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col(dbc.Button("Find Peaks", id="at-peaks-find-btn", color="primary"), width="auto"),
            dbc.Col(dbc.Button("Compute FWHM", id="at-peaks-fwhm-btn", color="secondary"), width="auto"),
            dbc.Col(dbc.Button("Integrate Peaks", id="at-peaks-int-btn", color="secondary"), width="auto"),
        ], className="g-2 mb-3"),
        html.Div(id="at-peaks-output"),
    ])


@callback(Output("at-peaks-output", "children"),
          Input("at-peaks-find-btn", "n_clicks"),
          Input("at-peaks-fwhm-btn", "n_clicks"),
          Input("at-peaks-int-btn", "n_clicks"),
          State("at-dataset", "value"),
          State("at-peaks-col", "value"),
          State("at-peaks-height", "value"),
          State("at-peaks-prom", "value"),
          State("at-peaks-dist", "value"),
          prevent_initial_call=True)
def do_peaks(find_n, fwhm_n, int_n, ds_name, col, height, prom, dist):
    arr = _get_arr(ds_name, col)
    if arr is None:
        return dbc.Alert("No data.", color="warning")
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        peaks_result = pk_find_peaks(arr, height=height, prominence=prom, distance=int(dist or 5))
        peak_idx = peaks_result.get("peaks", np.array([]))

        if triggered == "at-peaks-find-btn":
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=arr, mode="lines", name="Signal"))
            if len(peak_idx):
                fig.add_trace(go.Scatter(x=peak_idx, y=arr[peak_idx], mode="markers",
                                         name="Peaks", marker={"size": 8, "color": "#e0a3a3"}))
            fig.update_layout(template="plotly_dark", title=f"Peak Detection ({len(peak_idx)} peaks)", paper_bgcolor="rgba(0,0,0,0)")
            return html.Div([dcc.Graph(figure=fig, style={"height": "380px"}),
                             html.Pre(str(peaks_result), className="result-box mt-2")])

        if triggered == "at-peaks-fwhm-btn":
            fwhm = compute_fwhm(arr, peaks=peak_idx)
            return html.Pre(str(fwhm), className="result-box")

        if triggered == "at-peaks-int-btn":
            integrals = integrate_peaks(arr, peaks=peak_idx)
            return html.Pre(str(integrals), className="result-box")

    except Exception as e:
        return dbc.Alert(str(e), color="danger")
    return dash.no_update


# ── Statistical Tests tab ─────────────────────────────────────────────────────

def _tests_tab(col_opts, num_cols):
    return html.Div([
        dbc.Tabs([
            dbc.Tab(label="t-Tests", tab_id="tt-ttest"),
            dbc.Tab(label="ANOVA", tab_id="tt-anova"),
            dbc.Tab(label="Non-parametric", tab_id="tt-nonparam"),
            dbc.Tab(label="Correlation", tab_id="tt-corr"),
        ], id="tt-tabs", active_tab="tt-ttest"),
        html.Div(id="tt-content", className="mt-3"),
        html.Div(id="tt-output"),
    ])


@callback(Output("tt-content", "children"),
          Input("tt-tabs", "active_tab"),
          State("at-dataset", "value"))
def tt_tab(tab, ds_name):
    data = state.get_dataset(ds_name) if ds_name else None
    num_cols = _num_cols(data)
    col_opts = [{"label": c, "value": c} for c in num_cols]

    if tab == "tt-ttest":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Test type"),
                         dcc.Dropdown(id="tt-ttest-type",
                                      options=[{"label": t, "value": t}
                                               for t in ["One-sample", "Two-sample (independent)", "Paired"]],
                                      value="One-sample", clearable=False)], width=4),
                dbc.Col([dbc.Label("Column 1"), dcc.Dropdown(id="tt-col1", options=col_opts,
                                                              value=num_cols[0] if num_cols else None, clearable=False)], width=3),
                dbc.Col([dbc.Label("Column 2 / μ₀"),
                         dbc.Input(id="tt-col2-or-mu", placeholder="col or μ₀=0", size="sm")], width=3),
            ], className="g-2 mb-2"),
            dbc.Button("Run t-Test", id="tt-ttest-btn", color="primary"),
        ])
    if tab == "tt-anova":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Groups (columns)"), dcc.Dropdown(id="tt-anova-cols", options=col_opts,
                                                                      value=num_cols[:3] if len(num_cols) >= 3 else num_cols,
                                                                      multi=True)], width=6),
            ], className="g-2 mb-2"),
            dbc.Button("One-way ANOVA", id="tt-anova-btn", color="primary"),
        ])
    if tab == "tt-nonparam":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Test"),
                         dcc.Dropdown(id="tt-nonparam-type",
                                      options=[{"label": t, "value": t}
                                               for t in ["Mann-Whitney U", "Wilcoxon Signed-Rank", "Kruskal-Wallis"]],
                                      value="Mann-Whitney U", clearable=False)], width=4),
                dbc.Col([dbc.Label("Group 1"), dcc.Dropdown(id="tt-np-g1", options=col_opts,
                                                             value=num_cols[0] if num_cols else None, clearable=False)], width=3),
                dbc.Col([dbc.Label("Group 2"), dcc.Dropdown(id="tt-np-g2", options=col_opts,
                                                             value=num_cols[1] if len(num_cols) > 1 else None, clearable=True)], width=3),
            ], className="g-2 mb-2"),
            dbc.Button("Run Test", id="tt-nonparam-btn", color="primary"),
        ])
    if tab == "tt-corr":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Column 1"), dcc.Dropdown(id="tt-corr-c1", options=col_opts,
                                                              value=num_cols[0] if num_cols else None, clearable=False)], width=3),
                dbc.Col([dbc.Label("Column 2"), dcc.Dropdown(id="tt-corr-c2", options=col_opts,
                                                              value=num_cols[1] if len(num_cols) > 1 else None, clearable=False)], width=3),
                dbc.Col([dbc.Label("Method"),
                         dcc.Dropdown(id="tt-corr-method",
                                      options=[{"label": "Pearson", "value": "pearson"},
                                               {"label": "Spearman", "value": "spearman"}],
                                      value="pearson", clearable=False)], width=3),
            ], className="g-2 mb-2"),
            dbc.Button("Compute Correlation", id="tt-corr-btn", color="primary"),
        ])
    return html.Div()


@callback(Output("tt-output", "children"),
          Input("tt-ttest-btn", "n_clicks"),
          Input("tt-anova-btn", "n_clicks"),
          Input("tt-nonparam-btn", "n_clicks"),
          Input("tt-corr-btn", "n_clicks"),
          State("at-dataset", "value"),
          State("tt-ttest-type", "value"),
          State("tt-col1", "value"),
          State("tt-col2-or-mu", "value"),
          State("tt-anova-cols", "value"),
          State("tt-nonparam-type", "value"),
          State("tt-np-g1", "value"),
          State("tt-np-g2", "value"),
          State("tt-corr-c1", "value"),
          State("tt-corr-c2", "value"),
          State("tt-corr-method", "value"),
          prevent_initial_call=True)
def run_test(t_n, a_n, np_n, c_n, ds_name, ttype, c1, c2_or_mu, anova_cols, np_type, np_g1, np_g2, cc1, cc2, corr_m):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    data = state.get_dataset(ds_name) if ds_name else None

    try:
        if triggered == "tt-ttest-btn":
            arr1 = _get_arr(ds_name, c1)
            if ttype == "One-sample":
                mu0 = float(c2_or_mu or 0)
                result = ttest_one_sample(arr1, mu=mu0)
            elif ttype == "Two-sample (independent)":
                arr2 = _get_arr(ds_name, c2_or_mu) if isinstance(data, pd.DataFrame) and c2_or_mu in (data.columns if isinstance(data, pd.DataFrame) else []) else None
                if arr2 is None:
                    return dbc.Alert("Second column not found.", color="warning")
                result = ttest_two_sample(arr1, arr2)
            else:
                arr2 = _get_arr(ds_name, c2_or_mu)
                if arr2 is None:
                    return dbc.Alert("Second column not found.", color="warning")
                result = ttest_paired(arr1, arr2)
            return html.Pre(str(result), className="result-box")

        if triggered == "tt-anova-btn":
            groups = [_get_arr(ds_name, c) for c in (anova_cols or []) if _get_arr(ds_name, c) is not None]
            result = anova_oneway(*groups)
            return html.Pre(str(result), className="result-box")

        if triggered == "tt-nonparam-btn":
            g1 = _get_arr(ds_name, np_g1)
            g2 = _get_arr(ds_name, np_g2) if np_g2 else None
            if np_type == "Mann-Whitney U":
                result = mannwhitney_u(g1, g2)
            elif np_type == "Wilcoxon Signed-Rank":
                result = wilcoxon_signed_rank(g1, g2)
            else:
                groups = [g1] + ([g2] if g2 is not None else [])
                result = kruskal_wallis(*groups)
            return html.Pre(str(result), className="result-box")

        if triggered == "tt-corr-btn":
            a1 = _get_arr(ds_name, cc1)
            a2 = _get_arr(ds_name, cc2)
            if corr_m == "pearson":
                result = pearson_correlation(a1, a2)
            else:
                result = spearman_correlation(a1, a2)
            return html.Pre(str(result), className="result-box")

    except Exception as e:
        return dbc.Alert(str(e), color="danger")
    return dash.no_update


# ── Helpers ───────────────────────────────────────────────────────────────────

def _num_cols(data) -> list:
    if isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number]).columns.tolist()
    return []


def _get_arr(ds_name, col):
    if not ds_name:
        return None
    data = state.get_dataset(ds_name)
    if data is None:
        return None
    if isinstance(data, pd.DataFrame) and col and col in data.columns:
        return data[col].dropna().values.astype(float)
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        return data.flatten().astype(float)
    return None
