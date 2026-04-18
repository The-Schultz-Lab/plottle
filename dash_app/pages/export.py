"""Export Results page.

Sections
--------
1. Dataset Export  — download the current dataset as CSV / JSON / Excel / Parquet
2. Analysis Export — download stored analysis results as JSON
3. Session Save    — serialise full session to JSON and download
4. Session Load    — upload a saved session file to restore state
"""

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
import base64

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dash_app import state

dash.register_page(__name__, path="/export", title="Export — Plottle", name="Export")


def layout(**kwargs):
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Export Results", className="page-title"),
                    html.P(
                        "Download datasets, analysis results, and session state.",
                        className="page-caption",
                    ),
                ],
                className="page-header",
            ),
            # Section 1 — Dataset export
            html.H3("Export Dataset", className="mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Dataset"),
                            dcc.Dropdown(
                                id="ex-ds-select",
                                options=[
                                    {"label": n, "value": n} for n in state.get_dataset_names()
                                ],
                                value=state._STATE.get("current_dataset"),
                                clearable=False,
                                placeholder="Select dataset…",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Format"),
                            dcc.Dropdown(
                                id="ex-ds-format",
                                options=[
                                    {"label": "CSV", "value": "csv"},
                                    {"label": "JSON", "value": "json"},
                                    {"label": "Excel (.xlsx)", "value": "xlsx"},
                                    {"label": "Parquet", "value": "parquet"},
                                    {"label": "NumPy (.npy)", "value": "npy"},
                                ],
                                value="csv",
                                clearable=False,
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Download Dataset",
                            id="ex-ds-btn",
                            color="primary",
                            className="mt-4 w-100",
                        ),
                        width=3,
                    ),
                ],
                className="g-3 mb-2",
            ),
            dcc.Download(id="ex-ds-download"),
            dbc.Alert(id="ex-ds-feedback", is_open=False, dismissable=True, className="mt-2"),
            html.Hr(),
            # Section 2 — Analysis results
            html.H3("Export Analysis Results", className="mb-2"),
            _analysis_section(),
            dcc.Download(id="ex-analysis-download"),
            html.Hr(),
            # Section 3 — Session save/load
            html.H3("Session Save / Load", className="mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Save Session (JSON)",
                            id="ex-session-save-btn",
                            color="primary",
                            className="w-100",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Restore from file"),
                            dcc.Upload(
                                id="ex-session-upload",
                                children=html.Div(["Drop or ", html.A("select session .json")]),
                                accept=".json",
                                style={
                                    "border": "2px dashed var(--border)",
                                    "borderRadius": "6px",
                                    "padding": "0.4rem",
                                    "textAlign": "center",
                                    "cursor": "pointer",
                                    "background": "var(--bg-secondary)",
                                },
                            ),
                        ],
                        width=4,
                    ),
                ],
                className="g-3 mb-2",
            ),
            dcc.Download(id="ex-session-download"),
            html.Div(id="ex-session-feedback", className="mt-2"),
        ]
    )


def _analysis_section():
    results = state.get_analysis_results()
    if not results:
        return dbc.Alert(
            "No analysis results yet. Run analyses in the Analysis Tools page first.", color="info"
        )

    rows = [
        html.Tr(
            [
                html.Td(str(i + 1)),
                html.Td(r.get("type", "—")),
                html.Td(r.get("dataset", "—")),
                html.Td(r.get("timestamp", "—")[:19]),
            ]
        )
        for i, r in enumerate(results)
    ]

    return html.Div(
        [
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [html.Th("#"), html.Th("Type"), html.Th("Dataset"), html.Th("Time")]
                        )
                    ),
                    html.Tbody(rows),
                ],
                bordered=True,
                hover=True,
                size="sm",
                className="mb-2",
            ),
            dbc.Button("Download All as JSON", id="ex-analysis-btn", color="secondary"),
        ]
    )


# ── Dataset download ──────────────────────────────────────────────────────────


@callback(
    Output("ex-ds-download", "data"),
    Output("ex-ds-feedback", "children"),
    Output("ex-ds-feedback", "is_open"),
    Output("ex-ds-feedback", "color"),
    Input("ex-ds-btn", "n_clicks"),
    State("ex-ds-select", "value"),
    State("ex-ds-format", "value"),
    prevent_initial_call=True,
)
def download_dataset(n, ds_name, fmt):
    if not n or not ds_name:
        return dash.no_update, dash.no_update, False, "info"

    data = state.get_dataset(ds_name)
    if data is None:
        return dash.no_update, "Dataset not found.", True, "danger"

    stem = Path(ds_name).stem
    try:
        if fmt == "csv" and isinstance(data, pd.DataFrame):
            return (
                dcc.send_data_frame(data.to_csv, f"{stem}.csv", index=False),
                "",
                False,
                "success",
            )

        if fmt == "json" and isinstance(data, pd.DataFrame):
            return (
                dcc.send_data_frame(data.to_json, f"{stem}.json", orient="records"),
                "",
                False,
                "success",
            )

        if fmt == "xlsx" and isinstance(data, pd.DataFrame):
            return (
                dcc.send_data_frame(data.to_excel, f"{stem}.xlsx", index=False),
                "",
                False,
                "success",
            )

        if fmt == "parquet" and isinstance(data, pd.DataFrame):
            buf = io.BytesIO()
            data.to_parquet(buf, index=False)
            buf.seek(0)
            return dcc.send_bytes(buf.read(), f"{stem}.parquet"), "", False, "success"

        if fmt == "npy":
            arr = data.values if isinstance(data, pd.DataFrame) else data
            buf = io.BytesIO()
            np.save(buf, arr)
            buf.seek(0)
            return dcc.send_bytes(buf.read(), f"{stem}.npy"), "", False, "success"

        # Fallback: CSV
        if isinstance(data, pd.DataFrame):
            return (
                dcc.send_data_frame(data.to_csv, f"{stem}.csv", index=False),
                "",
                False,
                "success",
            )

        return dash.no_update, f"Cannot export {type(data).__name__} as {fmt}.", True, "warning"

    except Exception as e:
        return dash.no_update, str(e), True, "danger"


# ── Analysis download ─────────────────────────────────────────────────────────


@callback(
    Output("ex-analysis-download", "data"),
    Input("ex-analysis-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_analysis(n):
    if not n:
        return dash.no_update
    results = state.get_analysis_results()
    json_str = json.dumps(results, indent=2, default=str)
    return dcc.send_string(json_str, "plottle_analysis.json")


# ── Session save ──────────────────────────────────────────────────────────────


@callback(
    Output("ex-session-download", "data"),
    Output("ex-session-feedback", "children", allow_duplicate=True),
    Input("ex-session-save-btn", "n_clicks"),
    prevent_initial_call=True,
)
def save_session(n):
    if not n:
        return dash.no_update, dash.no_update

    summary = state.get_session_summary()
    session = {
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "current_dataset": summary["current_dataset"],
        "num_datasets": summary["num_datasets"],
        "dataset_names": summary["dataset_names"],
        "analysis_results": state.get_analysis_results(),
        "datasets": {},
    }

    # Serialize datasets
    for name in summary["dataset_names"]:
        data = state.get_dataset(name)
        if isinstance(data, pd.DataFrame):
            session["datasets"][name] = {
                "__type__": "DataFrame",
                "__data__": data.to_json(orient="split"),
            }
        elif isinstance(data, np.ndarray):
            import pickle

            session["datasets"][name] = {
                "__type__": "ndarray",
                "__data__": base64.b64encode(pickle.dumps(data)).decode("utf-8"),
            }

    json_str = json.dumps(session, indent=2, default=str)
    return (
        dcc.send_string(json_str, "plottle_session.json"),
        dbc.Alert("Session saved.", color="success", dismissable=True),
    )


# ── Session load ──────────────────────────────────────────────────────────────


@callback(
    Output("ex-session-feedback", "children"),
    Input("ex-session-upload", "contents"),
    State("ex-session-upload", "filename"),
    prevent_initial_call=True,
)
def load_session(contents, filename):
    if contents is None:
        return dash.no_update

    _ct, content_string = contents.split(",")
    decoded = base64.b64decode(content_string).decode("utf-8")

    try:
        session = json.loads(decoded)
        import pickle

        for name, payload in session.get("datasets", {}).items():
            if isinstance(payload, dict) and "__type__" in payload:
                if payload["__type__"] == "DataFrame":
                    data = pd.read_json(io.StringIO(payload["__data__"]), orient="split")
                elif payload["__type__"] == "ndarray":
                    data = pickle.loads(base64.b64decode(payload["__data__"]))
                else:
                    data = payload["__data__"]
            else:
                data = payload
            state.add_dataset(name, data)

        if session.get("current_dataset"):
            state.set_current_dataset(session["current_dataset"])

        for result in session.get("analysis_results", []):
            state.add_analysis_result(result)

        return dbc.Alert(
            f"Session restored: {session.get('num_datasets', 0)} datasets, "
            f"{len(session.get('analysis_results', []))} analysis results.",
            color="success",
            dismissable=True,
        )
    except Exception as e:
        return dbc.Alert(f"Failed to load session: {e}", color="danger", dismissable=True)
