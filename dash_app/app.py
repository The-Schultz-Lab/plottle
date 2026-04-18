"""Plottle — Dash App Entry Point.

Replaces the Streamlit ``plottle/Home.py`` with a full Dash multi-page
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

# Make repo root importable so ``plottle.*`` resolve correctly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from dash_app.themes import DEFAULT_THEME, THEMES as _THEMES

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
            html.Img(src=_logo_img_src, className="sidebar-logo-img")
            if _logo_img_src
            else html.Div(),
            className="sidebar-logo-wrap",
        ),
        html.Div(
            [
                html.Button(
                    "",
                    id=f"theme-btn-{slug}",
                    className=f"theme-swatch theme-swatch-{slug}",
                    title=label,
                    **{"data-theme-slug": slug},
                )
                for slug, label in _THEMES
            ],
            className="theme-switcher",
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

# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="theme-store", storage_type="local", data=DEFAULT_THEME),
        html.Div(id="_theme-target", children="", style={"display": "none"}),
        dbc.Toast(
            id="global-toast",
            header="Plottle",
            is_open=False,
            dismissable=True,
            duration=3000,
            style={"position": "fixed", "top": "1rem", "right": "1rem", "zIndex": 9999},
        ),
        sidebar,
        html.Main(
            dash.page_container,
            className="main-content",
            id="page-content",
        ),
    ],
    className="app-wrapper",
)


# ── Shutdown route ────────────────────────────────────────────────────────────

_SHUTDOWN_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Plottle — stopped</title>
<style>
  body { font-family: 'Nunito', 'Segoe UI', sans-serif; display: flex;
         align-items: center; justify-content: center; height: 100vh;
         margin: 0; background: #1a1a2e; color: #ccc; }
  .box { text-align: center; }
  .box h2 { color: #e0a3a3; margin-bottom: 0.4rem; }
  .box p  { color: #888; font-size: 0.95rem; }
</style>
</head><body><div class="box">
<h2>Plottle has stopped</h2>
<p>The server has shut down. You can close this tab.</p>
</div></body></html>"""


@app.server.route("/shutdown")
def _shutdown_page():
    """Serve a goodbye page then kill the process."""
    from flask import Response

    threading.Timer(0.6, lambda: os._exit(0)).start()
    return Response(_SHUTDOWN_HTML, mimetype="text/html")


# ── Exit callback ─────────────────────────────────────────────────────────────


@app.callback(
    Output("url", "pathname"),
    Input("exit-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_exit(n_clicks):
    """Redirect to /shutdown, which serves a goodbye page then kills the process."""
    return "/shutdown"


# ── Theme callbacks ───────────────────────────────────────────────────────────

app.clientside_callback(
    """
    function(theme) {
        var t = theme || 'nccu-dark';
        document.documentElement.setAttribute('data-theme', t);
        document.querySelectorAll('.theme-swatch').forEach(function(b) {
            b.classList.toggle('swatch-active', b.getAttribute('data-theme-slug') === t);
        });
        return t;
    }
    """,
    Output("_theme-target", "children"),
    Input("theme-store", "data"),
)


@app.callback(
    Output("theme-store", "data"),
    [Input(f"theme-btn-{slug}", "n_clicks") for slug, _ in _THEMES],
    prevent_initial_call=True,
)
def on_swatch_click(*_):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    for slug, _ in _THEMES:
        if f"theme-btn-{slug}" == btn_id:
            return slug
    return dash.no_update


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050, host="127.0.0.1")
