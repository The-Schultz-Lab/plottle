"""Data Tools page.

10 tabs of DataFrame transformation operations:
  Formula Column  — add a new column via Python expression
  Normalize       — min-max or z-score normalization
  Transpose       — transpose rows/columns
  Pivot / Melt    — reshape wide ↔ long
  Filter          — filter rows by column value
  Sort            — sort by one or more columns
  Merge           — join two datasets
  Fill / Drop NaN — impute or drop missing values
  Resample        — time-series resampling
  Rolling         — rolling-window aggregations
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
from modules.data_tools import (
    add_formula_column,
    fill_missing_values,
    filter_rows,
    melt_dataframe,
    merge_dataframes,
    normalize_column,
    pivot_dataframe,
    resample_dataframe,
    rolling_aggregate,
    sort_dataframe,
    transpose_dataframe,
)

dash.register_page(__name__, path="/analyze-data-tools", title="Data Tools — Plottle", name="Data Tools")


def layout(**kwargs):
    names = state.get_dataset_names()
    ds_opts = [{"label": n, "value": n} for n in names]
    current = state._STATE.get("current_dataset")

    return html.Div(
        [
            html.Div(
                [
                    html.H1("Data Tools", className="page-title"),
                    html.P("Transform and reshape DataFrames.", className="page-caption"),
                ],
                className="page-header",
            ),
            dbc.Row(
                dbc.Col(
                    [dbc.Label("Dataset"), dcc.Dropdown(id="dt-dataset", options=ds_opts, value=current,
                                                         clearable=False, placeholder="Select dataset…")],
                    width=5,
                ),
                className="mb-3",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(label="Formula Column", tab_id="dt-formula"),
                    dbc.Tab(label="Normalize", tab_id="dt-norm"),
                    dbc.Tab(label="Transpose", tab_id="dt-transpose"),
                    dbc.Tab(label="Pivot / Melt", tab_id="dt-pivot"),
                    dbc.Tab(label="Filter", tab_id="dt-filter"),
                    dbc.Tab(label="Sort", tab_id="dt-sort"),
                    dbc.Tab(label="Merge", tab_id="dt-merge"),
                    dbc.Tab(label="Fill / Drop NaN", tab_id="dt-nan"),
                    dbc.Tab(label="Resample", tab_id="dt-resample"),
                    dbc.Tab(label="Rolling", tab_id="dt-rolling"),
                ],
                id="dt-tabs",
                active_tab="dt-formula",
            ),
            html.Div(id="dt-tab-content", className="tab-content"),
            html.Hr(),
            html.Div(id="dt-result-section"),
        ]
    )


@callback(
    Output("dt-tab-content", "children"),
    Input("dt-tabs", "active_tab"),
    Input("dt-dataset", "value"),
)
def render_tab(tab, ds_name):
    df = _get_df(ds_name)
    cols = list(df.columns) if df is not None else []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist() if df is not None else []
    col_opts = [{"label": c, "value": c} for c in cols]
    num_opts = [{"label": c, "value": c} for c in num_cols]
    all_ds = [{"label": n, "value": n} for n in state.get_dataset_names()]

    if tab == "dt-formula":
        return html.Div([
            html.P("Add a new column computed from existing columns using Python / NumPy.", className="text-muted-sm mb-2"),
            dbc.Row([
                dbc.Col([dbc.Label("New column name"), dbc.Input(id="dt-f-name", placeholder="result", size="sm")], width=3),
                dbc.Col([dbc.Label("Expression (use df['col'] or col_name)"),
                         dbc.Input(id="dt-f-expr", placeholder="df['x'] ** 2 + np.log(df['y'])", size="sm")], width=6),
            ], className="g-2 mb-3"),
            _save_row("dt-formula"),
        ])

    if tab == "dt-norm":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Column(s)"), dcc.Dropdown(id="dt-n-cols", options=num_opts, multi=True)], width=4),
                dbc.Col([dbc.Label("Method"),
                         dcc.Dropdown(id="dt-n-method",
                                      options=[{"label": "Min-Max (0–1)", "value": "minmax"},
                                               {"label": "Z-score", "value": "zscore"},
                                               {"label": "Max abs", "value": "maxabs"}],
                                      value="minmax", clearable=False)], width=3),
            ], className="g-2 mb-3"),
            _save_row("dt-norm"),
        ])

    if tab == "dt-transpose":
        return html.Div([
            html.P("Transpose the DataFrame (rows ↔ columns).", className="text-muted-sm mb-3"),
            _save_row("dt-transpose"),
        ])

    if tab == "dt-pivot":
        return html.Div([
            dbc.Tabs([
                dbc.Tab(label="Pivot", tab_id="dt-pv"),
                dbc.Tab(label="Melt", tab_id="dt-melt"),
            ], id="dt-pivot-sub", active_tab="dt-pv"),
            dbc.Row([
                dbc.Col([dbc.Label("Index column"), dcc.Dropdown(id="dt-pv-idx", options=col_opts)], width=3),
                dbc.Col([dbc.Label("Columns column"), dcc.Dropdown(id="dt-pv-cols", options=col_opts)], width=3),
                dbc.Col([dbc.Label("Values column"), dcc.Dropdown(id="dt-pv-vals", options=num_opts)], width=3),
                dbc.Col([dbc.Label("ID vars (melt)"), dcc.Dropdown(id="dt-melt-ids", options=col_opts, multi=True)], width=3),
            ], className="g-2 mt-2 mb-3"),
            _save_row("dt-pivot"),
        ])

    if tab == "dt-filter":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Column"), dcc.Dropdown(id="dt-flt-col", options=col_opts, clearable=False)], width=3),
                dbc.Col([dbc.Label("Operator"),
                         dcc.Dropdown(id="dt-flt-op",
                                      options=[{"label": o, "value": o} for o in [">", ">=", "<", "<=", "==", "!=", "contains"]],
                                      value=">", clearable=False)], width=2),
                dbc.Col([dbc.Label("Value"), dbc.Input(id="dt-flt-val", size="sm")], width=3),
            ], className="g-2 mb-3"),
            _save_row("dt-filter"),
        ])

    if tab == "dt-sort":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Sort by"), dcc.Dropdown(id="dt-srt-cols", options=col_opts, multi=True)], width=4),
                dbc.Col([dbc.Label("Order"),
                         dcc.Dropdown(id="dt-srt-asc",
                                      options=[{"label": "Ascending", "value": True},
                                               {"label": "Descending", "value": False}],
                                      value=True, clearable=False)], width=3),
            ], className="g-2 mb-3"),
            _save_row("dt-sort"),
        ])

    if tab == "dt-merge":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Right dataset"), dcc.Dropdown(id="dt-mrg-right", options=all_ds)], width=4),
                dbc.Col([dbc.Label("Join type"),
                         dcc.Dropdown(id="dt-mrg-how",
                                      options=[{"label": t, "value": t} for t in ["inner", "outer", "left", "right"]],
                                      value="inner", clearable=False)], width=3),
                dbc.Col([dbc.Label("On column"), dcc.Dropdown(id="dt-mrg-on", options=col_opts)], width=3),
            ], className="g-2 mb-3"),
            _save_row("dt-merge"),
        ])

    if tab == "dt-nan":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Action"),
                         dcc.Dropdown(id="dt-nan-action",
                                      options=[{"label": "Drop rows with NaN", "value": "drop_rows"},
                                               {"label": "Drop columns with NaN", "value": "drop_cols"},
                                               {"label": "Fill with mean", "value": "fill_mean"},
                                               {"label": "Fill with median", "value": "fill_median"},
                                               {"label": "Fill with constant", "value": "fill_const"},
                                               {"label": "Forward fill", "value": "ffill"},
                                               {"label": "Backward fill", "value": "bfill"}],
                                      value="drop_rows", clearable=False)], width=4),
                dbc.Col([dbc.Label("Fill value (if fill_const)"), dbc.Input(id="dt-nan-val", type="number", value=0, size="sm")], width=3),
            ], className="g-2 mb-3"),
            _save_row("dt-nan"),
        ])

    if tab == "dt-resample":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Datetime / index column"), dcc.Dropdown(id="dt-rs-col", options=col_opts)], width=3),
                dbc.Col([dbc.Label("New size (rows)"), dbc.Input(id="dt-rs-size", type="number", value=100, min=2, size="sm")], width=2),
                dbc.Col([dbc.Label("Method"),
                         dcc.Dropdown(id="dt-rs-method",
                                      options=[{"label": m, "value": m} for m in ["linear", "nearest", "cubic"]],
                                      value="linear", clearable=False)], width=3),
            ], className="g-2 mb-3"),
            _save_row("dt-resample"),
        ])

    if tab == "dt-rolling":
        return html.Div([
            dbc.Row([
                dbc.Col([dbc.Label("Column"), dcc.Dropdown(id="dt-roll-col", options=num_opts)], width=3),
                dbc.Col([dbc.Label("Window size"), dbc.Input(id="dt-roll-win", type="number", value=5, min=2, size="sm")], width=2),
                dbc.Col([dbc.Label("Aggregation"),
                         dcc.Dropdown(id="dt-roll-agg",
                                      options=[{"label": a, "value": a} for a in ["mean", "sum", "std", "min", "max", "median"]],
                                      value="mean", clearable=False)], width=3),
                dbc.Col([dbc.Label("New column name"), dbc.Input(id="dt-roll-name", placeholder="rolling_mean", size="sm")], width=3),
            ], className="g-2 mb-3"),
            _save_row("dt-rolling"),
        ])

    return html.Div()


def _save_row(tab_id: str):
    return dbc.Row(
        [
            dbc.Col(dbc.Button("Preview", id=f"{tab_id}-preview-btn", color="secondary"), width="auto"),
            dbc.Col(dbc.Button("Save as new dataset", id=f"{tab_id}-save-btn", color="primary"), width="auto"),
            dbc.Col([dbc.Label("Name"), dbc.Input(id=f"{tab_id}-save-name", placeholder="result_dataset", size="sm")], width=3),
        ],
        className="g-2",
    )


# ── One mega-callback routing to each operation ───────────────────────────────

_TAB_BTNS = {
    "dt-formula": ("dt-formula-preview-btn", "dt-formula-save-btn"),
    "dt-norm": ("dt-norm-preview-btn", "dt-norm-save-btn"),
    "dt-transpose": ("dt-transpose-preview-btn", "dt-transpose-save-btn"),
    "dt-pivot": ("dt-pivot-preview-btn", "dt-pivot-save-btn"),
    "dt-filter": ("dt-filter-preview-btn", "dt-filter-save-btn"),
    "dt-sort": ("dt-sort-preview-btn", "dt-sort-save-btn"),
    "dt-merge": ("dt-merge-preview-btn", "dt-merge-save-btn"),
    "dt-nan": ("dt-nan-preview-btn", "dt-nan-save-btn"),
    "dt-resample": ("dt-resample-preview-btn", "dt-resample-save-btn"),
    "dt-rolling": ("dt-rolling-preview-btn", "dt-rolling-save-btn"),
}


@callback(
    Output("dt-result-section", "children"),
    # All preview + save buttons
    Input("dt-formula-preview-btn", "n_clicks"),
    Input("dt-formula-save-btn", "n_clicks"),
    Input("dt-norm-preview-btn", "n_clicks"),
    Input("dt-norm-save-btn", "n_clicks"),
    Input("dt-transpose-preview-btn", "n_clicks"),
    Input("dt-transpose-save-btn", "n_clicks"),
    Input("dt-pivot-preview-btn", "n_clicks"),
    Input("dt-pivot-save-btn", "n_clicks"),
    Input("dt-filter-preview-btn", "n_clicks"),
    Input("dt-filter-save-btn", "n_clicks"),
    Input("dt-sort-preview-btn", "n_clicks"),
    Input("dt-sort-save-btn", "n_clicks"),
    Input("dt-merge-preview-btn", "n_clicks"),
    Input("dt-merge-save-btn", "n_clicks"),
    Input("dt-nan-preview-btn", "n_clicks"),
    Input("dt-nan-save-btn", "n_clicks"),
    Input("dt-resample-preview-btn", "n_clicks"),
    Input("dt-resample-save-btn", "n_clicks"),
    Input("dt-rolling-preview-btn", "n_clicks"),
    Input("dt-rolling-save-btn", "n_clicks"),
    # States
    State("dt-dataset", "value"),
    State("dt-tabs", "active_tab"),
    # Formula states
    State("dt-f-name", "value"),
    State("dt-f-expr", "value"),
    # Normalize states
    State("dt-n-cols", "value"),
    State("dt-n-method", "value"),
    # Pivot/Melt states
    State("dt-pv-idx", "value"),
    State("dt-pv-cols", "value"),
    State("dt-pv-vals", "value"),
    State("dt-melt-ids", "value"),
    State("dt-pivot-sub", "active_tab"),
    # Filter states
    State("dt-flt-col", "value"),
    State("dt-flt-op", "value"),
    State("dt-flt-val", "value"),
    # Sort
    State("dt-srt-cols", "value"),
    State("dt-srt-asc", "value"),
    # Merge
    State("dt-mrg-right", "value"),
    State("dt-mrg-how", "value"),
    State("dt-mrg-on", "value"),
    # NaN
    State("dt-nan-action", "value"),
    State("dt-nan-val", "value"),
    # Resample
    State("dt-rs-col", "value"),
    State("dt-rs-size", "value"),
    State("dt-rs-method", "value"),
    # Rolling
    State("dt-roll-col", "value"),
    State("dt-roll-win", "value"),
    State("dt-roll-agg", "value"),
    State("dt-roll-name", "value"),
    # Save names
    State("dt-formula-save-name", "value"),
    State("dt-norm-save-name", "value"),
    State("dt-transpose-save-name", "value"),
    State("dt-pivot-save-name", "value"),
    State("dt-filter-save-name", "value"),
    State("dt-sort-save-name", "value"),
    State("dt-merge-save-name", "value"),
    State("dt-nan-save-name", "value"),
    State("dt-resample-save-name", "value"),
    State("dt-rolling-save-name", "value"),
    prevent_initial_call=True,
)
def apply_operation(
    # preview/save n_clicks (ignored, just triggers)
    *args,
):
    # Unpack args by position
    (f_prev, f_save, n_prev, n_save, tr_prev, tr_save, pv_prev, pv_save,
     flt_prev, flt_save, srt_prev, srt_save, mrg_prev, mrg_save,
     nan_prev, nan_save, rs_prev, rs_save, roll_prev, roll_save,
     ds_name, active_tab,
     f_name, f_expr,
     n_cols, n_method,
     pv_idx, pv_cols, pv_vals, melt_ids, pivot_sub,
     flt_col, flt_op, flt_val,
     srt_cols, srt_asc,
     mrg_right, mrg_how, mrg_on,
     nan_action, nan_val,
     rs_col, rs_size, rs_method,
     roll_col, roll_win, roll_agg, roll_name,
     f_sname, n_sname, tr_sname, pv_sname, flt_sname, srt_sname,
     mrg_sname, nan_sname, rs_sname, roll_sname) = args

    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    is_save = triggered.endswith("-save-btn")

    df = _get_df(ds_name)
    if df is None:
        return dbc.Alert("No DataFrame loaded.", color="warning")

    result_df = None
    save_name = ""

    try:
        if "formula" in triggered:
            result_df = add_formula_column(df, column_name=f_name or "new_col", expression=f_expr or "0")
            save_name = f_sname or "formula_result"
        elif "norm" in triggered:
            result_df = df.copy()
            for c in (n_cols or df.select_dtypes(include=[np.number]).columns.tolist()):
                result_df = normalize_column(result_df, column=c, method=n_method or "minmax")
            save_name = n_sname or "normalized"
        elif "transpose" in triggered:
            result_df = transpose_dataframe(df)
            save_name = tr_sname or "transposed"
        elif "pivot" in triggered:
            if pivot_sub == "dt-melt":
                result_df = melt_dataframe(df, id_vars=melt_ids or [])
            else:
                result_df = pivot_dataframe(df, index=pv_idx, columns=pv_cols, values=pv_vals)
            save_name = pv_sname or "pivoted"
        elif "filter" in triggered:
            result_df = filter_rows(df, column=flt_col, operator=flt_op or ">", value=flt_val or 0)
            save_name = flt_sname or "filtered"
        elif "sort" in triggered:
            result_df = sort_dataframe(df, by=srt_cols or [df.columns[0]], ascending=srt_asc)
            save_name = srt_sname or "sorted"
        elif "merge" in triggered:
            right_df = _get_df(mrg_right)
            if right_df is None:
                return dbc.Alert("Right dataset not found or not a DataFrame.", color="warning")
            result_df = merge_dataframes(df, right_df, how=mrg_how or "inner", on=mrg_on)
            save_name = mrg_sname or "merged"
        elif "nan" in triggered:
            result_df = fill_missing_values(df, method=nan_action or "drop_rows", fill_value=float(nan_val or 0))
            save_name = nan_sname or "nan_handled"
        elif "resample" in triggered:
            result_df = resample_dataframe(df, new_size=int(rs_size or 100), method=rs_method or "linear")
            save_name = rs_sname or "resampled"
        elif "rolling" in triggered:
            result_df = rolling_aggregate(df, column=roll_col, window=int(roll_win or 5),
                                          agg=roll_agg or "mean", new_col_name=roll_name or "rolling")
            save_name = roll_sname or "rolling_result"

    except Exception as e:
        return dbc.Alert(str(e), color="danger")

    if result_df is None:
        return dbc.Alert("Operation did not produce a result.", color="warning")

    if is_save:
        state.add_dataset(save_name or "result", result_df, metadata={"source": "data_tools"})
        feedback = dbc.Alert(f"Saved as '{save_name}'.", color="success", dismissable=True)
    else:
        feedback = html.Div()

    return html.Div([
        feedback,
        html.H5(f"Result: {result_df.shape[0]} rows × {result_df.shape[1]} columns"),
        _df_preview(result_df),
    ])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_df(ds_name):
    if not ds_name:
        return None
    data = state.get_dataset(ds_name)
    return data if isinstance(data, pd.DataFrame) else None


def _df_preview(df: pd.DataFrame):
    preview = df.head(20)
    return dash_table.DataTable(
        data=preview.to_dict("records"),
        columns=[{"name": c, "id": c} for c in preview.columns],
        style_table={"overflowX": "auto", "maxHeight": "320px", "overflowY": "auto"},
        style_cell={"backgroundColor": "var(--bg-primary)", "color": "var(--text-primary)",
                    "border": "1px solid var(--border)", "fontSize": "0.82rem"},
        style_header={"backgroundColor": "var(--bg-secondary)", "fontWeight": "600"},
        page_size=20,
        fixed_rows={"headers": True},
    )
