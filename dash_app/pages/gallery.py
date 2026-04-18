"""Gallery page.

Displays pre-rendered example figures from docs/gallery/ in a 3-column card grid.
Clicking "Use this config" loads the plot config into Quick Plot.
"""

import json
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

dash.register_page(__name__, path="/gallery", title="Gallery — Plottle", name="Gallery")

_APP_ROOT = Path(__file__).resolve().parents[2]
_GALLERY_DIR = _APP_ROOT / "docs" / "gallery"
_MANIFEST = _GALLERY_DIR / "manifest.json"

_LIB_COLORS = {
    "matplotlib": "#e0a3a3",
    "seaborn": "#56b4e9",
    "plotly": "#2ca02c",
}


def _load_manifest() -> list:
    if not _MANIFEST.exists():
        return []
    try:
        with open(_MANIFEST) as f:
            return json.load(f)
    except Exception:
        return []


def layout(**kwargs):
    manifest = _load_manifest()
    if not manifest:
        return html.Div(
            [
                html.Div(
                    [
                        html.H1("Gallery", className="page-title"),
                        html.P("Browse pre-rendered example figures.", className="page-caption"),
                    ],
                    className="page-header",
                ),
                dbc.Alert(
                    [
                        "No gallery images found. Run ",
                        html.Code("python generate_gallery.py"),
                        " from the repo root to generate example figures.",
                    ],
                    color="info",
                ),
            ]
        )

    # Group by library
    groups: dict = {}
    for item in manifest:
        lib = item.get("library", "other").lower()
        groups.setdefault(lib, []).append(item)

    sections = []
    for lib, items in groups.items():
        color = _LIB_COLORS.get(lib, "#aaaaaa")
        header = html.H3(lib.capitalize(), style={"color": color}, className="mt-4 mb-3")

        cards = []
        for item in items:
            img_path = item.get("image", "")
            img_src = f"/assets/../docs/gallery/{img_path}" if img_path else ""
            # Serve from /docs/gallery/ — Dash serves static files from assets/
            # Copy images to assets/gallery/ or serve via Flask route
            img_src = f"/{img_path}" if img_path else ""

            card = dbc.Col(
                html.Div(
                    [
                        html.Img(
                            src=img_src,
                            style={
                                "width": "100%",
                                "height": "160px",
                                "objectFit": "cover",
                                "borderRadius": "4px",
                                "marginBottom": "0.5rem",
                            },
                            onerror="this.style.display='none'",
                        )
                        if img_src
                        else html.Div(
                            html.P(
                                "No image",
                                style={
                                    "textAlign": "center",
                                    "color": "var(--text-muted)",
                                    "lineHeight": "160px",
                                    "height": "160px",
                                },
                            ),
                        ),
                        html.Div(item.get("title", ""), className="gallery-card-title"),
                        html.Div(
                            item.get("description", ""),
                            className="gallery-card-lib text-muted-sm mb-2",
                        ),
                        dbc.Button(
                            "Use this config",
                            id={"type": "gallery-use-btn", "index": item.get("plot_type", "")},
                            size="sm",
                            color="outline-secondary",
                            className="w-100",
                        )
                        if item.get("plot_type")
                        else html.Div(),
                    ],
                    className="gallery-card",
                ),
                width=4,
                className="mb-3",
            )
            cards.append(card)

        # Rows of 3
        rows = []
        for i in range(0, len(cards), 3):
            rows.append(dbc.Row(cards[i : i + 3], className="g-3"))

        sections.extend([header, *rows])

    return html.Div(
        [
            html.Div(
                [
                    html.H1("Gallery", className="page-title"),
                    html.P(
                        f"{len(manifest)} example figures grouped by library.",
                        className="page-caption",
                    ),
                ],
                className="page-header",
            ),
            html.Div(id="gallery-feedback"),
            html.Div(sections),
        ]
    )


@callback(
    Output("gallery-feedback", "children"),
    Input({"type": "gallery-use-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def use_config(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return dash.no_update
    plot_type = ctx.triggered_id["index"]
    if not plot_type:
        return dash.no_update
    return dbc.Alert(
        [
            f"Config for '{plot_type}' noted. ",
            dbc.Button(
                "Open Quick Plot →",
                href="/plot-basic",
                color="link",
                className="p-0",
                style={"color": "var(--accent)"},
            ),
            " and select this plot type.",
        ],
        color="success",
        dismissable=True,
    )
