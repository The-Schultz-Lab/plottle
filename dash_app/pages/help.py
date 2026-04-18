"""Help page.

5 accordion sections mirroring the Streamlit version:
  Getting Started, Plot Types (26 overview), Analysis Tools,
  File Formats, Tips & Tricks
"""

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

dash.register_page(__name__, path="/help", title="Help — Plottle", name="Help")

_PLOT_TYPES_TABLE = [
    ("Histogram", "Matplotlib", "Single-variable distribution"),
    ("Line Plot", "Matplotlib", "Ordered / time-series data"),
    ("Scatter Plot", "Matplotlib", "Two-variable relationship"),
    ("Bar Chart", "Matplotlib", "Category comparisons"),
    ("Heatmap", "Matplotlib", "2D matrix data"),
    ("Contour Plot", "Matplotlib", "Level curves of 2D arrays"),
    ("Waterfall Plot", "Matplotlib", "Cumulative change visualization"),
    ("Dual Axis Plot", "Matplotlib", "Two y-scales on one chart"),
    ("Broken Axis Plot", "Matplotlib", "Skip outlier gaps"),
    ("Z-Colored Scatter", "Matplotlib", "Colour scatter by third variable"),
    ("Bubble Chart", "Matplotlib", "Scatter with size encoding"),
    ("Polar Plot", "Matplotlib", "Angular / radial data"),
    ("2D Histogram", "Matplotlib", "Joint density of two variables"),
    ("Scatter with Regression", "Matplotlib", "Scatter + trend line"),
    ("Residual Plot", "Matplotlib", "Fit residuals"),
    ("Inset Plot", "Matplotlib", "Zoom-in subplot"),
    ("Distribution Plot", "Seaborn", "KDE + histogram overlay"),
    ("Box Plot", "Seaborn", "Quartile distribution"),
    ("Regression Plot", "Seaborn", "Scatter + CI band"),
    ("Pair Plot", "Seaborn", "All pairwise relationships"),
    ("Interactive Histogram", "Plotly", "Hover-enabled histogram"),
    ("Interactive Scatter", "Plotly", "Zoom/pan scatter"),
    ("Interactive Line", "Plotly", "Hover-enabled line"),
    ("Interactive Heatmap", "Plotly", "Colour-mapped matrix"),
    ("Interactive 3D Surface", "Plotly", "Rotating 3D surface"),
    ("Interactive 3D Scatter", "Plotly", "Rotating 3D scatter"),
    ("Interactive Ternary", "Plotly", "3-component composition"),
]


def layout(**kwargs):
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Help", className="page-title"),
                    html.P("Documentation for Plottle v2.0.0", className="page-caption"),
                ],
                className="page-header",
            ),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        _getting_started(),
                        title="Getting Started",
                        item_id="gs",
                    ),
                    dbc.AccordionItem(
                        _plot_types_section(),
                        title="Plot Types (26 total)",
                        item_id="pt",
                    ),
                    dbc.AccordionItem(
                        _analysis_tools_section(),
                        title="Analysis Tools",
                        item_id="at",
                    ),
                    dbc.AccordionItem(
                        _file_formats_section(),
                        title="File Formats",
                        item_id="ff",
                    ),
                    dbc.AccordionItem(
                        _tips_section(),
                        title="Tips & Tricks",
                        item_id="tt",
                    ),
                ],
                start_collapsed=False,
                active_item="gs",
            ),
        ]
    )


def _getting_started():
    return html.Div(
        [
            html.Ol(
                [
                    html.Li(
                        [
                            html.Strong("Upload Data"),
                            " — Go to ",
                            html.Em("Data Upload"),
                            " and load a CSV, Excel, NumPy, or other supported file. "
                            "Or click one of the 10 built-in example datasets.",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Plot"),
                            " — Navigate to ",
                            dbc.Button(
                                "Plot → Basic",
                                href="/plot-basic",
                                color="link",
                                className="p-0",
                                style={"color": "var(--accent)"},
                            ),
                            " and select a plot type from the 26 available. Configure in the left panel.",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Analyse"),
                            " — ",
                            dbc.Button(
                                "Analyze → Single",
                                href="/analyze-single",
                                color="link",
                                className="p-0",
                                style={"color": "var(--accent)"},
                            ),
                            " has descriptive statistics, curve fitting, signal processing, peak detection, and statistical tests.",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Export"),
                            " — ",
                            dbc.Button(
                                "Export",
                                href="/export",
                                color="link",
                                className="p-0",
                                style={"color": "var(--accent)"},
                            ),
                            " downloads datasets as CSV/JSON/Excel, analysis results as JSON, or the full session.",
                        ]
                    ),
                    html.Li(
                        [
                            html.Strong("Multi-Plot"),
                            " — ",
                            dbc.Button(
                                "Plot → Multiplot",
                                href="/plot-multiplot",
                                color="link",
                                className="p-0",
                                style={"color": "var(--accent)"},
                            ),
                            " places up to 16 charts in a 1×1 to 4×4 grid.",
                        ]
                    ),
                ],
                style={"lineHeight": "2"},
            ),
        ]
    )


def _plot_types_section():
    rows = [
        html.Tr([html.Td(name), html.Td(lib, style={"color": _lib_color(lib)}), html.Td(desc)])
        for name, lib, desc in _PLOT_TYPES_TABLE
    ]
    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Plot Type"), html.Th("Library"), html.Th("Use Case")])),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        size="sm",
        responsive=True,
    )


def _analysis_tools_section():
    sections = [
        (
            "Statistics",
            "Descriptive stats (mean, median, std, quartiles, IQR, range). Shapiro-Wilk normality test.",
        ),
        (
            "Distribution Fitting",
            "Fit parametric distributions (normal, exponential, gamma, lognormal, beta) and overlay PDF.",
        ),
        (
            "Curve Fitting",
            "Linear, polynomial (degree 1–10), exponential, and custom expression fits with R² and residuals.",
        ),
        (
            "Optimization",
            "Single-variable minimization (Nelder-Mead, BFGS, Powell) and root-finding over a range.",
        ),
        (
            "Linear Algebra",
            "Eigenvalue decomposition, linear system solve (Ax=b), matrix decompositions (SVD/QR/LU/Cholesky).",
        ),
        (
            "Signal Processing",
            "Moving-average/Savitzky-Golay/Gaussian smoothing, lowpass/highpass/bandpass/bandstop filters, FFT, derivative, polynomial/rolling-ball/ALS baseline, interpolation.",
        ),
        (
            "Peak Analysis",
            "SciPy-based peak detection with height/prominence/distance thresholds, FWHM computation, peak integration, multi-peak Gaussian fitting.",
        ),
        (
            "Statistical Tests",
            "One/two-sample and paired t-tests, one-way ANOVA, Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Pearson and Spearman correlations.",
        ),
    ]
    return html.Dl(
        [
            item
            for name, desc in sections
            for item in [html.Dt(html.Strong(name)), html.Dd(desc, className="ms-4 mb-2")]
        ]
    )


def _file_formats_section():
    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Category"), html.Th("Extensions"), html.Th("Notes")])),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td("Tabular"),
                            html.Td(".csv, .tsv, .xlsx, .xls, .json, .parquet"),
                            html.Td("Auto-detected via extension"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("NumPy"),
                            html.Td(".npy, .npz"),
                            html.Td("Multi-array .npz → dict of arrays"),
                        ]
                    ),
                    html.Tr(
                        [html.Td("Python"), html.Td(".pkl"), html.Td("Arbitrary pickled objects")]
                    ),
                    html.Tr(
                        [
                            html.Td("Spectroscopy"),
                            html.Td(".jdx, .dx, .spc, .asc"),
                            html.Td("JCAMP-DX, Thermo SPC, ASCII"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Scientific"),
                            html.Td(".h5, .hdf5, .nc, .cdf, .mzml"),
                            html.Td("HDF5, NetCDF, mzML mass spec"),
                        ]
                    ),
                ]
            ),
        ],
        bordered=True,
        hover=True,
        size="sm",
    )


def _tips_section():
    tips = [
        (
            "Session management",
            "Datasets and analysis results persist across all pages during a session. "
            "Use Export → Save Session to preserve everything as a JSON file between restarts.",
        ),
        (
            "Large files",
            "Files >50 MB or >10,000 rows get a downsampled preview automatically. "
            "The full data is still loaded and available for analysis.",
        ),
        (
            "Choosing a plot type",
            "Histogram → distributions. Scatter → correlations. Line → time series. "
            "Box/Violin → group comparisons. Heatmap → 2D matrices.",
        ),
        (
            "Publication figures",
            "Check 'Publication style' in Quick Plot's configuration panel for serif font and clean lines. "
            "Export at 300 DPI from the Export page, or use SVG for vector graphics.",
        ),
        (
            "Batch analysis",
            "Use Analyze → Batch to run the same operation across multiple loaded datasets simultaneously. "
            "Save results as a CSV table.",
        ),
        (
            "Plugins",
            "Place plugin_*.py files in the plugins/ directory at the repo root. "
            "Plugins that expose get_plot_types() or get_analysis_tools() are auto-discovered.",
        ),
        (
            "CLI usage",
            "Plottle also runs headlessly: python cli.py plot --input data.csv --type scatter --output fig.png, "
            "or python cli.py stats --input data.csv --column temperature.",
        ),
    ]
    return dbc.Accordion(
        [
            dbc.AccordionItem(html.P(desc, style={"fontSize": "0.88rem"}), title=title)
            for title, desc in tips
        ],
        flush=True,
        start_collapsed=True,
    )


def _lib_color(lib: str) -> str:
    return {"Matplotlib": "#e0a3a3", "Seaborn": "#56b4e9", "Plotly": "#2ca02c"}.get(lib, "#aaa")
