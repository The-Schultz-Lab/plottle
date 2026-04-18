"""Settings page.

Persistent defaults and named presets saved to config.json.

Sections
--------
1. Default Plot Style — font, size, colour, grid, legend, palette
2. Presets            — save / load / delete named configurations
3. Config file info   — path and raw JSON viewer
"""

import json
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plottle.utils.user_settings import (
    delete_preset,
    get_config_path,
    get_defaults,
    list_presets,
    load_preset,
    save_defaults,
    save_preset,
)
from plottle.utils.plot_config import COLOR_PALETTE_NAMES, _FONT_OPTIONS
from dash_app.themes import DEFAULT_THEME, THEMES as _THEMES

dash.register_page(__name__, path="/settings", title="Settings — Plottle", name="Settings")

_FONT_OPTIONS_LIST = (
    _FONT_OPTIONS
    if isinstance(_FONT_OPTIONS, list)
    else [
        "sans-serif",
        "serif",
        "monospace",
        "Helvetica",
        "Arial",
        "Times New Roman",
        "DejaVu Sans",
    ]
)

_PALETTE_OPTIONS = (
    COLOR_PALETTE_NAMES
    if isinstance(COLOR_PALETTE_NAMES, list)
    else ["Default", "Okabe-Ito", "Wong", "Tol Muted", "Pastel", "Vibrant"]
)


def layout(**kwargs):
    config_path = _safe_config_path()

    return html.Div(
        [
            html.Div(
                [
                    html.H1("Settings", className="page-title"),
                    html.P(
                        [
                            "Persistent defaults and presets are saved to ",
                            html.Code(str(config_path)),
                            ". Changes take effect the next time you open Quick Plot.",
                        ],
                        className="page-caption",
                    ),
                ],
                className="page-header",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(label="Default Style", tab_id="st-defaults"),
                    dbc.Tab(label="Presets", tab_id="st-presets"),
                    dbc.Tab(label="Config File", tab_id="st-config"),
                ],
                id="st-tabs",
                active_tab="st-defaults",
            ),
            html.Div(id="st-tab-content", className="tab-content"),
        ]
    )


@callback(Output("st-tab-content", "children"), Input("st-tabs", "active_tab"))
def render_tab(tab):
    if tab == "st-defaults":
        return _defaults_tab()
    if tab == "st-presets":
        return _presets_tab()
    return _config_tab()


# ── Theme callbacks ───────────────────────────────────────────────────────────


@callback(
    Output("st-theme-select", "value"),
    Input("theme-store", "data"),
)
def sync_theme_dropdown(theme):
    return theme or DEFAULT_THEME


@callback(
    Output("theme-store", "data", allow_duplicate=True),
    Input("st-theme-select", "value"),
    prevent_initial_call=True,
)
def update_theme_from_dropdown(theme):
    return theme or dash.no_update


# ── Defaults tab ──────────────────────────────────────────────────────────────


def _defaults_tab():
    defaults = _safe_get_defaults()
    return html.Div(
        [
            html.Div(
                [
                    html.P("APP APPEARANCE", className="config-section-title"),
                    dbc.Label("Color scheme"),
                    dcc.Dropdown(
                        id="st-theme-select",
                        options=[{"label": label, "value": slug} for slug, label in _THEMES],
                        value=DEFAULT_THEME,
                        clearable=False,
                    ),
                    html.P(
                        "Theme is stored in your browser and applies immediately. "
                        "You can also switch themes with the colored swatches in the sidebar.",
                        className="text-muted-sm mt-1 mb-0",
                    ),
                ],
                className="mb-4",
            ),
            html.P(
                "These values are applied as initial defaults in Quick Plot.",
                className="text-muted-sm mb-3",
            ),
            dbc.Row(
                [
                    # Typography
                    dbc.Col(
                        [
                            html.P("TYPOGRAPHY", className="config-section-title"),
                            dbc.Label("Font family"),
                            dcc.Dropdown(
                                id="st-fontfamily",
                                options=[{"label": f, "value": f} for f in _FONT_OPTIONS_LIST],
                                value=defaults.get("fontfamily", "sans-serif"),
                                clearable=False,
                            ),
                            dbc.Label("Font size", className="mt-2"),
                            dcc.Slider(
                                id="st-fontsize",
                                min=6,
                                max=24,
                                step=1,
                                value=defaults.get("fontsize", 10),
                                marks={6: "6", 10: "10", 14: "14", 18: "18", 24: "24"},
                                tooltip={"placement": "bottom"},
                            ),
                            dbc.Label("Font color", className="mt-2"),
                            dbc.Input(
                                id="st-fontcolor",
                                type="color",
                                value=defaults.get("fontcolor", "#ffffff"),
                                style={"height": "36px", "padding": "2px"},
                            ),
                        ],
                        width=4,
                    ),
                    # Line & grid
                    dbc.Col(
                        [
                            html.P("LINE & GRID", className="config-section-title"),
                            dbc.Label("Line width"),
                            dcc.Slider(
                                id="st-linewidth",
                                min=0.5,
                                max=5.0,
                                step=0.25,
                                value=defaults.get("linewidth", 1.5),
                                marks={1: "1", 2: "2", 3: "3", 5: "5"},
                                tooltip={"placement": "bottom"},
                            ),
                            dbc.Checklist(
                                id="st-grid",
                                options=[{"label": " Show grid by default", "value": "grid"}],
                                value=["grid"] if defaults.get("grid", True) else [],
                                inputStyle={"marginRight": "6px"},
                                className="mt-2",
                            ),
                            dbc.Label("Grid line style", className="mt-1"),
                            dcc.Dropdown(
                                id="st-grid-ls",
                                options=[
                                    {"label": "Solid", "value": "-"},
                                    {"label": "Dashed", "value": "--"},
                                    {"label": "Dotted", "value": ":"},
                                    {"label": "Dash-dot", "value": "-."},
                                ],
                                value=defaults.get("grid_linestyle", "--"),
                                clearable=False,
                            ),
                        ],
                        width=4,
                    ),
                    # Color & legend
                    dbc.Col(
                        [
                            html.P("COLORS & LEGEND", className="config-section-title"),
                            dbc.Label("Color palette"),
                            dcc.Dropdown(
                                id="st-palette",
                                options=[{"label": p, "value": p} for p in _PALETTE_OPTIONS],
                                value=defaults.get("color_palette", "Default"),
                                clearable=False,
                            ),
                            dbc.Label("Legend position", className="mt-2"),
                            dcc.Dropdown(
                                id="st-legend-pos",
                                options=[
                                    {"label": "Best", "value": "best"},
                                    {"label": "Upper right", "value": "upper right"},
                                    {"label": "Upper left", "value": "upper left"},
                                    {"label": "Lower right", "value": "lower right"},
                                    {"label": "Lower left", "value": "lower left"},
                                ],
                                value=defaults.get("legend_position", "best"),
                                clearable=False,
                            ),
                        ],
                        width=4,
                    ),
                ],
                className="g-3 mb-3",
            ),
            dbc.Button(
                "Save Defaults", id="st-save-defaults-btn", color="primary", className="mb-3"
            ),
            html.Div(id="st-defaults-feedback"),
        ]
    )


@callback(
    Output("st-defaults-feedback", "children"),
    Input("st-save-defaults-btn", "n_clicks"),
    State("st-fontfamily", "value"),
    State("st-fontsize", "value"),
    State("st-fontcolor", "value"),
    State("st-linewidth", "value"),
    State("st-grid", "value"),
    State("st-grid-ls", "value"),
    State("st-palette", "value"),
    State("st-legend-pos", "value"),
    prevent_initial_call=True,
)
def save_defaults_cb(
    n, fontfamily, fontsize, fontcolor, linewidth, grid, grid_ls, palette, legend_pos
):
    if not n:
        return dash.no_update
    new_defaults = {
        "fontfamily": fontfamily,
        "fontsize": int(fontsize or 10),
        "fontcolor": fontcolor,
        "linewidth": float(linewidth or 1.5),
        "grid": "grid" in (grid or []),
        "grid_linestyle": grid_ls,
        "color_palette": palette,
        "legend_position": legend_pos,
    }
    try:
        save_defaults(new_defaults)
        return dbc.Alert("Defaults saved to config.json.", color="success", dismissable=True)
    except Exception as e:
        return dbc.Alert(str(e), color="danger", dismissable=True)


# ── Presets tab ───────────────────────────────────────────────────────────────


def _presets_tab():
    presets = _safe_list_presets()
    preset_opts = [{"label": p, "value": p} for p in presets]

    return html.Div(
        [
            html.H5("Save Current Defaults as Preset"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Preset name"),
                            dbc.Input(
                                id="st-preset-name", placeholder="publication_style", size="sm"
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Save Preset",
                            id="st-preset-save-btn",
                            color="primary",
                            className="mt-4",
                        ),
                        width=2,
                    ),
                ],
                className="g-2 mb-3",
            ),
            html.Hr(),
            html.H5("Load / Delete Preset"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Preset"),
                            dcc.Dropdown(
                                id="st-preset-select",
                                options=preset_opts,
                                placeholder="Select preset…",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Load", id="st-preset-load-btn", color="secondary", className="mt-4"
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Delete", id="st-preset-del-btn", color="danger", className="mt-4"
                        ),
                        width=2,
                    ),
                ],
                className="g-2 mb-3",
            ),
            html.Div(id="st-preset-feedback"),
        ]
    )


@callback(
    Output("st-preset-feedback", "children"),
    Input("st-preset-save-btn", "n_clicks"),
    Input("st-preset-load-btn", "n_clicks"),
    Input("st-preset-del-btn", "n_clicks"),
    State("st-preset-name", "value"),
    State("st-preset-select", "value"),
    prevent_initial_call=True,
)
def manage_presets(save_n, load_n, del_n, name, selected):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered == "st-preset-save-btn":
        if not name:
            return dbc.Alert("Enter a preset name.", color="warning")
        try:
            save_preset(name, _safe_get_defaults())
            return dbc.Alert(f"Preset '{name}' saved.", color="success", dismissable=True)
        except Exception as e:
            return dbc.Alert(str(e), color="danger")

    if triggered == "st-preset-load-btn":
        if not selected:
            return dbc.Alert("Select a preset.", color="warning")
        try:
            p = load_preset(selected)
            return html.Div(
                [
                    dbc.Alert(f"Loaded preset '{selected}'.", color="success", dismissable=True),
                    html.Pre(json.dumps(p, indent=2), className="result-box"),
                ]
            )
        except Exception as e:
            return dbc.Alert(str(e), color="danger")

    if triggered == "st-preset-del-btn":
        if not selected:
            return dbc.Alert("Select a preset.", color="warning")
        try:
            delete_preset(selected)
            return dbc.Alert(f"Preset '{selected}' deleted.", color="success", dismissable=True)
        except Exception as e:
            return dbc.Alert(str(e), color="danger")

    return dash.no_update


# ── Config file tab ───────────────────────────────────────────────────────────


def _config_tab():
    config_path = _safe_config_path()
    raw = ""
    try:
        if config_path.exists():
            raw = config_path.read_text(encoding="utf-8")
    except Exception:
        pass

    return html.Div(
        [
            html.P(
                ["Config file location: ", html.Code(str(config_path))],
                className="text-muted-sm mb-3",
            ),
            html.H5("Raw config.json"),
            html.Pre(raw or "(file does not exist yet)", className="result-box"),
            dbc.Button(
                "Clear config.json", id="st-clear-config-btn", color="danger", className="mt-2"
            ),
            html.Div(id="st-config-feedback", className="mt-2"),
        ]
    )


@callback(
    Output("st-config-feedback", "children"),
    Input("st-clear-config-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_config(n):
    if not n:
        return dash.no_update
    try:
        config_path = _safe_config_path()
        if config_path.exists():
            config_path.unlink()
            return dbc.Alert("config.json deleted.", color="success", dismissable=True)
        return dbc.Alert("config.json does not exist.", color="info")
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


# ── Safe wrappers ─────────────────────────────────────────────────────────────


def _safe_get_defaults() -> dict:
    try:
        return get_defaults()
    except Exception:
        return {}


def _safe_list_presets() -> list:
    try:
        return list_presets()
    except Exception:
        return []


def _safe_config_path() -> Path:
    try:
        return get_config_path()
    except Exception:
        return Path.cwd() / "config.json"
