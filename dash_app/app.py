"""Plottle — Dash App Entry Point.

Replaces the Streamlit ``modules/Home.py`` with a full Dash multi-page
application that preserves all features and the NCCU dark colour scheme.

Run
---
    # from repo root:
    python dash_app/app.py

    # or with Gunicorn in production:
    gunicorn "dash_app.app:server"
"""

import os
import sys
import threading
from pathlib import Path

# Make repo root importable so ``modules.*`` resolve correctly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

# ── App instance ──────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=str(Path(__file__).parent / "pages"),
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Plottle",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server  # expose Flask server for Gunicorn


# ── Logo (embed as base64 so it works regardless of static-file routing) ─────

_logo_path = _REPO_ROOT / "logo.png"


def _embed_logo() -> str:
    try:
        import base64
        return "data:image/png;base64," + base64.b64encode(_logo_path.read_bytes()).decode()
    except Exception:
        return ""


_logo_img_src = _embed_logo()


# ── Sidebar helpers ───────────────────────────────────────────────────────────

def _nav_link(label: str, href: str) -> dbc.NavLink:
    return dbc.NavLink(label, href=href, active="exact", className="sidebar-link")


def _nav_section(title: str, links: list, id_suffix: str) -> html.Details:
    return html.Details(
        [
            html.Summary(title, className="nav-section-summary"),
            html.Div(links, className="nav-section-body"),
        ],
        id=f"nav-group-{id_suffix}",
        className="nav-group",
        open=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

sidebar = html.Div(
    [
        html.Div(
            html.Img(src=_logo_img_src, className="sidebar-logo-img") if _logo_img_src else html.Div(),
            className="sidebar-logo-wrap",
        ),
        html.Nav(
            [
                _nav_link("Home", "/"),
                _nav_link("Data Upload", "/data-upload"),
                html.Hr(className="sidebar-divider"),
                _nav_section(
                    "Plot",
                    [
                        _nav_link("Basic", "/plot-basic"),
                        _nav_link("Multiplot", "/plot-multiplot"),
                        html.P("Advanced", className="nav-subsection-label"),
                        _nav_link("↳ Advanced Plotting", "/plot-advanced"),
                        _nav_link("↳ Spectroscopy", "/plot-spectroscopy"),
                        _nav_link("↳ Molecular Viz", "/plot-molecular-viz"),
                    ],
                    "plot",
                ),
                _nav_section(
                    "Analyze",
                    [
                        _nav_link("Single", "/analyze-single"),
                        _nav_link("Batch", "/analyze-batch"),
                        _nav_link("Data Tools", "/analyze-data-tools"),
                    ],
                    "analyze",
                ),
                html.Hr(className="sidebar-divider"),
                _nav_link("Export", "/export"),
                _nav_link("Gallery", "/gallery"),
                _nav_link("Help", "/help"),
                _nav_link("Settings", "/settings"),
                html.Hr(className="sidebar-divider"),
                # ── Exit button ───────────────────────────────────────────────
                html.Div(
                    dbc.Button(
                        "⏻  Exit",
                        id="exit-btn",
                        color="danger",
                        outline=True,
                        size="sm",
                        className="w-100",
                        style={"fontSize": "0.82rem", "letterSpacing": "0.03em"},
                    ),
                    style={"padding": "0 0.5rem 0.5rem"},
                ),
                # ─────────────────────────────────────────────────────────────
                html.Div(
                    [
                        html.P(
                            "Scientific data visualization and analysis toolkit "
                            "developed at North Carolina Central University.",
                            className="sidebar-about-text",
                        ),
                        html.A(
                            "NCCU Schultz Lab →",
                            href="https://github.com/The-Schultz-Lab",
                            target="_blank",
                            className="sidebar-about-link",
                        ),
                    ],
                    className="sidebar-about",
                ),
            ],
        ),
    ],
    className="sidebar",
    id="sidebar",
)

# ── Exit confirmation modal ───────────────────────────────────────────────────

exit_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Shut down Plottle?")),
        dbc.ModalBody(
            "This will stop the server. You will need to re-run "
            "launch_dash.bat to start it again."
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Cancel", id="exit-cancel-btn", color="secondary", className="me-2"),
                dbc.Button("Shut Down", id="exit-confirm-btn", color="danger"),
            ]
        ),
    ],
    id="exit-modal",
    is_open=False,
    centered=True,
    backdrop="static",
)

# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dbc.Toast(
            id="global-toast",
            header="Plottle",
            is_open=False,
            dismissable=True,
            duration=3000,
            style={"position": "fixed", "top": "1rem", "right": "1rem", "zIndex": 9999},
        ),
        exit_modal,
        # Invisible placeholder — the confirm callback writes "shutdown" here
        # so the browser shows a message before the process exits.
        html.Div(id="exit-shutdown-placeholder", style={"display": "none"}),
        sidebar,
        html.Main(
            dash.page_container,
            className="main-content",
            id="page-content",
        ),
    ],
    className="app-wrapper",
)


# ── Exit callbacks ────────────────────────────────────────────────────────────

@app.callback(
    Output("exit-modal", "is_open"),
    Input("exit-btn", "n_clicks"),
    Input("exit-cancel-btn", "n_clicks"),
    State("exit-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_exit_modal(open_n, cancel_n, is_open):
    """Open the modal when Exit is clicked; close it when Cancel is clicked."""
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered == "exit-btn":
        return True
    if triggered == "exit-cancel-btn":
        return False
    return is_open


@app.callback(
    Output("exit-shutdown-placeholder", "children"),
    Input("exit-confirm-btn", "n_clicks"),
    prevent_initial_call=True,
)
def shutdown_server(n_clicks):
    """Schedule process exit after a short delay so the response is sent first."""
    if n_clicks:
        threading.Timer(0.4, lambda: os._exit(0)).start()
    return dash.no_update


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050, host="127.0.0.1")
