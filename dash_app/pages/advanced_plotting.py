"""Advanced Plotting page.

Mirrors Streamlit page 5: Seaborn statistical + Plotly interactive features.

Sub-sections (tabs):
  Correlation Heatmap    — seaborn / Plotly correlation matrix
  Overlaid Distributions — multiple column distributions on one canvas
  Grouped Categorical    — box/violin/swarm with hue grouping
  3D Scatter             — Plotly 3D scatter with colour axis
  HTML Export            — export interactive Plotly chart as self-contained HTML
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
import plotly.express as px
import seaborn as sns
from dash import Input, Output, State, callback, dcc, html
import base64

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dash_app import state

dash.register_page(
    __name__, path="/plot-advanced", title="Advanced Plotting — Plottle", name="Advanced Plotting"
)


def layout(**kwargs):
    names = state.get_dataset_names()
    ds_opts = [{"label": n, "value": n} for n in names]
    current = state._STATE.get("current_dataset")

    return html.Div(
        [
            html.Div(
                [
                    html.H1("Advanced Plotting", className="page-title"),
                    html.P(
                        "Seaborn statistical and Plotly interactive visualizations.",
                        className="page-caption",
                    ),
                ],
                className="page-header",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label("Dataset"),
                        dcc.Dropdown(
                            id="ap-dataset",
                            options=ds_opts,
                            value=current,
                            clearable=False,
                            placeholder="Select dataset…",
                        ),
                    ],
                    width=5,
                ),
                className="mb-3",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(label="Correlation Heatmap", tab_id="tab-corr"),
                    dbc.Tab(label="Overlaid Distributions", tab_id="tab-dist"),
                    dbc.Tab(label="Grouped Categorical", tab_id="tab-cat"),
                    dbc.Tab(label="3D Scatter", tab_id="tab-3d"),
                    dbc.Tab(label="HTML Export", tab_id="tab-html"),
                ],
                id="ap-tabs",
                active_tab="tab-corr",
            ),
            html.Div(id="ap-tab-content", className="tab-content"),
        ]
    )


@callback(
    Output("ap-tab-content", "children"),
    Input("ap-tabs", "active_tab"),
    Input("ap-dataset", "value"),
)
def render_tab(tab, ds_name):
    data = state.get_dataset(ds_name) if ds_name else None
    df = data if isinstance(data, pd.DataFrame) else None
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist() if df is not None else []
    cat_cols = [c for c in (df.columns.tolist() if df is not None else []) if c not in num_cols]
    col_opts = [{"label": c, "value": c} for c in (num_cols)]
    all_opts = [{"label": c, "value": c} for c in (df.columns.tolist() if df is not None else [])]

    if tab == "tab-corr":
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Columns (leave empty for all numeric)"),
                                dcc.Dropdown(
                                    id="ap-corr-cols",
                                    options=col_opts,
                                    value=[],
                                    multi=True,
                                    placeholder="All numeric columns",
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Method"),
                                dcc.Dropdown(
                                    id="ap-corr-method",
                                    options=[
                                        {"label": m.capitalize(), "value": m}
                                        for m in ["pearson", "spearman", "kendall"]
                                    ],
                                    value="pearson",
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Style"),
                                dcc.Dropdown(
                                    id="ap-corr-style",
                                    options=[
                                        {"label": "Seaborn", "value": "seaborn"},
                                        {"label": "Plotly", "value": "plotly"},
                                    ],
                                    value="plotly",
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="g-2 mb-3",
                ),
                dbc.Button("Generate", id="ap-corr-btn", color="primary", className="mb-3"),
                html.Div(id="ap-corr-output", className="plot-container"),
            ]
        )

    if tab == "tab-dist":
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Columns to overlay"),
                                dcc.Dropdown(
                                    id="ap-dist-cols",
                                    options=col_opts,
                                    value=num_cols[:3],
                                    multi=True,
                                    placeholder="Select columns…",
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Plot type"),
                                dcc.Dropdown(
                                    id="ap-dist-type",
                                    options=[
                                        {"label": t, "value": t}
                                        for t in [
                                            "KDE + Histogram",
                                            "KDE only",
                                            "Histogram only",
                                            "Violin",
                                            "Box",
                                        ]
                                    ],
                                    value="KDE + Histogram",
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="g-2 mb-3",
                ),
                dbc.Button("Generate", id="ap-dist-btn", color="primary", className="mb-3"),
                html.Div(id="ap-dist-output", className="plot-container"),
            ]
        )

    if tab == "tab-cat":
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("X (category)"),
                                dcc.Dropdown(
                                    id="ap-cat-x",
                                    options=all_opts,
                                    value=cat_cols[0]
                                    if cat_cols
                                    else (all_opts[0]["value"] if all_opts else None),
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Y (numeric)"),
                                dcc.Dropdown(
                                    id="ap-cat-y",
                                    options=col_opts,
                                    value=num_cols[0] if num_cols else None,
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Hue (optional)"),
                                dcc.Dropdown(
                                    id="ap-cat-hue",
                                    options=[{"label": "None", "value": ""}] + all_opts,
                                    value="",
                                    clearable=True,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Type"),
                                dcc.Dropdown(
                                    id="ap-cat-type",
                                    options=[
                                        {"label": t, "value": t}
                                        for t in ["Box", "Violin", "Strip", "Swarm", "Bar"]
                                    ],
                                    value="Box",
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="g-2 mb-3",
                ),
                dbc.Button("Generate", id="ap-cat-btn", color="primary", className="mb-3"),
                html.Div(id="ap-cat-output", className="plot-container"),
            ]
        )

    if tab == "tab-3d":
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("X"),
                                dcc.Dropdown(
                                    id="ap-3d-x",
                                    options=col_opts,
                                    value=num_cols[0] if num_cols else None,
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Y"),
                                dcc.Dropdown(
                                    id="ap-3d-y",
                                    options=col_opts,
                                    value=num_cols[1] if len(num_cols) > 1 else None,
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Z"),
                                dcc.Dropdown(
                                    id="ap-3d-z",
                                    options=col_opts,
                                    value=num_cols[2] if len(num_cols) > 2 else None,
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Color"),
                                dcc.Dropdown(
                                    id="ap-3d-c",
                                    options=[{"label": "None", "value": ""}] + col_opts,
                                    value="",
                                    clearable=True,
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="g-2 mb-3",
                ),
                dbc.Button("Generate", id="ap-3d-btn", color="primary", className="mb-3"),
                html.Div(id="ap-3d-output", className="plot-container"),
            ]
        )

    if tab == "tab-html":
        return html.Div(
            [
                html.P(
                    "Generate any Plotly figure above, then download it as an interactive HTML file.",
                    className="text-muted-sm",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("X"),
                                dcc.Dropdown(
                                    id="ap-html-x",
                                    options=col_opts,
                                    value=num_cols[0] if num_cols else None,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Y"),
                                dcc.Dropdown(
                                    id="ap-html-y",
                                    options=col_opts,
                                    value=num_cols[1] if len(num_cols) > 1 else None,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Color"),
                                dcc.Dropdown(
                                    id="ap-html-c",
                                    options=[{"label": "None", "value": ""}] + all_opts,
                                    value="",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Plot type"),
                                dcc.Dropdown(
                                    id="ap-html-type",
                                    options=[
                                        {"label": t, "value": t}
                                        for t in ["Scatter", "Line", "Bar", "Histogram", "Box"]
                                    ],
                                    value="Scatter",
                                    clearable=False,
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="g-2 mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("Preview", id="ap-html-preview-btn", color="primary"),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button("Download HTML", id="ap-html-dl-btn", color="secondary"),
                            width="auto",
                        ),
                        dbc.Col(dcc.Download(id="ap-html-download"), width="auto"),
                    ],
                    className="g-2 mb-3",
                ),
                html.Div(id="ap-html-output", className="plot-container"),
            ]
        )

    return html.Div()


# ── Correlation heatmap ───────────────────────────────────────────────────────


@callback(
    Output("ap-corr-output", "children"),
    Input("ap-corr-btn", "n_clicks"),
    State("ap-dataset", "value"),
    State("ap-corr-cols", "value"),
    State("ap-corr-method", "value"),
    State("ap-corr-style", "value"),
    prevent_initial_call=True,
)
def gen_corr(n, ds_name, cols, method, style):
    df = _get_df(ds_name)
    if df is None:
        return dbc.Alert("No DataFrame loaded.", color="warning")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    use_cols = [c for c in (cols or []) if c in num_cols] or num_cols
    corr = df[use_cols].corr(method=method)

    if style == "plotly":
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"{method.capitalize()} Correlation Matrix",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        return dcc.Graph(figure=fig, style={"height": "500px"})

    # Seaborn
    fig, ax = plt.subplots(figsize=(max(6, len(use_cols) * 0.8), max(5, len(use_cols) * 0.7)))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#1b1b1b")
    ax.set_facecolor("#1b1b1b")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title(f"{method.capitalize()} Correlation Matrix")
    return _fig_to_img(fig)


# ── Overlaid distributions ────────────────────────────────────────────────────


@callback(
    Output("ap-dist-output", "children"),
    Input("ap-dist-btn", "n_clicks"),
    State("ap-dataset", "value"),
    State("ap-dist-cols", "value"),
    State("ap-dist-type", "value"),
    prevent_initial_call=True,
)
def gen_dist(n, ds_name, cols, dist_type):
    df = _get_df(ds_name)
    if df is None:
        return dbc.Alert("No DataFrame loaded.", color="warning")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    use_cols = [c for c in (cols or []) if c in num_cols] or num_cols[:4]

    if dist_type in ("Violin", "Box"):
        fig = px.violin(
            df[use_cols].melt(var_name="Column", value_name="Value"),
            y="Value",
            x="Column",
            box=(dist_type == "Box"),
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return dcc.Graph(figure=fig, style={"height": "450px"})

    # Matplotlib KDE / histogram
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1b1b1b")
    ax.set_facecolor("#1b1b1b")

    from scipy.stats import gaussian_kde

    colors = ["#e0a3a3", "#56b4e9", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for i, col in enumerate(use_cols):
        arr = df[col].dropna().values.astype(float)
        c = colors[i % len(colors)]
        if dist_type in ("KDE + Histogram", "Histogram only"):
            ax.hist(arr, bins=25, density=True, alpha=0.4, color=c, label=col)
        if dist_type in ("KDE + Histogram", "KDE only") and len(arr) > 1:
            try:
                kde = gaussian_kde(arr)
                xs = np.linspace(arr.min(), arr.max(), 300)
                ax.plot(
                    xs,
                    kde(xs),
                    color=c,
                    linewidth=2,
                    label=f"{col} KDE" if dist_type == "KDE only" else "",
                )
            except Exception:
                pass

    ax.legend(fontsize=9)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Overlaid Distributions")
    return _fig_to_img(fig)


# ── Grouped categorical ───────────────────────────────────────────────────────


@callback(
    Output("ap-cat-output", "children"),
    Input("ap-cat-btn", "n_clicks"),
    State("ap-dataset", "value"),
    State("ap-cat-x", "value"),
    State("ap-cat-y", "value"),
    State("ap-cat-hue", "value"),
    State("ap-cat-type", "value"),
    prevent_initial_call=True,
)
def gen_cat(n, ds_name, x_col, y_col, hue_col, plot_type):
    df = _get_df(ds_name)
    if df is None:
        return dbc.Alert("No DataFrame loaded.", color="warning")
    if not x_col or not y_col:
        return dbc.Alert("Select X and Y columns.", color="warning")
    hue = hue_col if hue_col and hue_col in df.columns else None

    try:
        if plot_type == "Box":
            fig = px.box(df, x=x_col, y=y_col, color=hue, template="plotly_dark")
        elif plot_type == "Violin":
            fig = px.violin(df, x=x_col, y=y_col, color=hue, box=True, template="plotly_dark")
        elif plot_type == "Strip":
            fig = px.strip(df, x=x_col, y=y_col, color=hue, template="plotly_dark")
        elif plot_type == "Swarm":
            plt.style.use("dark_background")
            f, ax = plt.subplots(figsize=(9, 5))
            f.patch.set_facecolor("#1b1b1b")
            ax.set_facecolor("#1b1b1b")
            sns.swarmplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax, size=4)
            ax.set_title(f"{y_col} by {x_col}")
            return _fig_to_img(f)
        else:  # Bar
            fig = px.bar(df, x=x_col, y=y_col, color=hue, barmode="group", template="plotly_dark")

        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return dcc.Graph(figure=fig, style={"height": "450px"})
    except Exception as exc:
        return dbc.Alert(f"Error: {exc}", color="danger")


# ── 3D Scatter ────────────────────────────────────────────────────────────────


@callback(
    Output("ap-3d-output", "children"),
    Input("ap-3d-btn", "n_clicks"),
    State("ap-dataset", "value"),
    State("ap-3d-x", "value"),
    State("ap-3d-y", "value"),
    State("ap-3d-z", "value"),
    State("ap-3d-c", "value"),
    prevent_initial_call=True,
)
def gen_3d(n, ds_name, xc, yc, zc, cc):
    df = _get_df(ds_name)
    if df is None:
        return dbc.Alert("No DataFrame loaded.", color="warning")
    if not (xc and yc and zc):
        return dbc.Alert("Select X, Y, and Z columns.", color="warning")
    color = cc if cc and cc in df.columns else None
    fig = px.scatter_3d(df, x=xc, y=yc, z=zc, color=color, template="plotly_dark", opacity=0.75)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    return dcc.Graph(figure=fig, style={"height": "600px"})


# ── HTML Export ───────────────────────────────────────────────────────────────


@callback(
    Output("ap-html-output", "children"),
    Input("ap-html-preview-btn", "n_clicks"),
    State("ap-dataset", "value"),
    State("ap-html-x", "value"),
    State("ap-html-y", "value"),
    State("ap-html-c", "value"),
    State("ap-html-type", "value"),
    prevent_initial_call=True,
)
def html_preview(n, ds_name, xc, yc, cc, ptype):
    return _build_html_fig(ds_name, xc, yc, cc, ptype)


@callback(
    Output("ap-html-download", "data"),
    Input("ap-html-dl-btn", "n_clicks"),
    State("ap-dataset", "value"),
    State("ap-html-x", "value"),
    State("ap-html-y", "value"),
    State("ap-html-c", "value"),
    State("ap-html-type", "value"),
    prevent_initial_call=True,
)
def html_download(n, ds_name, xc, yc, cc, ptype):
    fig = _get_plotly_fig(ds_name, xc, yc, cc, ptype)
    if fig is None:
        return dash.no_update
    html_str = fig.to_html(include_plotlyjs="cdn")
    return dcc.send_string(html_str, "plottle_export.html")


def _build_html_fig(ds_name, xc, yc, cc, ptype):
    fig = _get_plotly_fig(ds_name, xc, yc, cc, ptype)
    if fig is None:
        return dbc.Alert("No data or columns available.", color="warning")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    return dcc.Graph(figure=fig, style={"height": "450px"})


def _get_plotly_fig(ds_name, xc, yc, cc, ptype):
    df = _get_df(ds_name)
    if df is None:
        return None
    color = cc if cc and cc in df.columns else None
    try:
        if ptype == "Scatter":
            return px.scatter(df, x=xc, y=yc, color=color, template="plotly_dark")
        elif ptype == "Line":
            return px.line(df, x=xc, y=yc, color=color, template="plotly_dark")
        elif ptype == "Bar":
            return px.bar(df, x=xc, y=yc, color=color, template="plotly_dark")
        elif ptype == "Histogram":
            return px.histogram(df, x=yc, color=color, template="plotly_dark")
        elif ptype == "Box":
            return px.box(df, x=xc, y=yc, color=color, template="plotly_dark")
    except Exception:
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_df(ds_name):
    if not ds_name:
        return None
    data = state.get_dataset(ds_name)
    return data if isinstance(data, pd.DataFrame) else None


def _fig_to_img(fig) -> html.Img:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return html.Img(
        src=f"data:image/png;base64,{encoded}", style={"width": "100%", "borderRadius": "4px"}
    )
