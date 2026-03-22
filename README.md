# Plottle

[![Tests](https://github.com/The-Schultz-Lab/plottle/actions/workflows/tests.yml/badge.svg)](https://github.com/The-Schultz-Lab/plottle/actions/workflows/tests.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/The-Schultz-Lab/plottle/main)

A general-purpose Python toolkit for scientific data visualization and analysis, developed for research and teaching in computational science at North Carolina Central University.

## Overview

Plottle provides a unified interface for scientific data work:

- **Data I/O** — 18 file formats (CSV, Excel, TSV, JSON, Parquet, NumPy, Pickle, JCAMP-DX, HDF5, NetCDF, SPC, ASC, mzML/mzXML, and more)
- **Mathematical analysis** — statistics, curve fitting, signal processing, peak analysis, hypothesis testing, optimization, linear algebra
- **Multi-library plotting** — 26 plot types across Matplotlib (static), Seaborn (statistical), and Plotly (interactive)
- **14-page Streamlit GUI** — exploratory data analysis without writing code
- **CLI** — batch processing and scripted workflows
- **Plugin system** — drop `plugin_*.py` into `plugins/` for custom plot types and tools

## Quick Start

### 1 — Clone and enter the repository

```bash
git clone https://github.com/The-Schultz-Lab/plottle.git
cd plottle
```

### 2 — Create a virtual environment and install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 3 — Launch the GUI

```bash
# Windows (double-click or terminal)
launch.bat

# Any platform
streamlit run modules/Home.py
```

Open `http://localhost:8501` in your browser.

### 4 — Or use the CLI

```bash
python cli.py --help
python cli.py plot data.csv --plot scatter --x-column x --y-column y
python cli.py stats data.csv
```

## GUI Pages

| Page | Description |
| --- | --- |
| **Home** | Dashboard overview and help tabs |
| **1 — Data Upload** | Upload files in 18 formats; preview shape, column types, and summary statistics; batch folder import |
| **2 — Quick Plot** | 26 plot types with live style controls, annotation panel, and Convert to Plotly toggle |
| **3 — Analysis Tools** | 8 tabs: Statistics, Distribution, Curve Fit, Optimization, Linear Algebra, Signal Processing, Peak Analysis, Statistical Tests |
| **4 — Multi-Plot Dashboard** | Up to 4×4 grid layouts with axis sharing and combined PNG/PDF export |
| **5 — Advanced Plotting** | Seaborn statistical plots and Plotly interactive charts with HTML export |
| **7 — Export Results** | Export plots, data, and analyses; save/load full sessions; generate PDF reports |
| **8 — Gallery** | Pre-generated figure gallery with Load this config buttons |
| **9 — Data Tools** | 10 operation tabs: formula columns, normalization, transpose, pivot/melt, filter, sort, merge, fill/drop NaN, resample, rolling transforms |
| **10 — Spectroscopy** | IR/Raman, NMR, UV-Vis, Mass Spec tools; NIST WebBook spectral lookup by CAS number |
| **11 — Molecular Viz** | Upload Gaussian/ORCA/Molden output files; 3D molecular structure and vibrational mode display |
| **12 — Batch Analysis** | Batch statistics, curve fitting, and peak analysis with workflow presets |
| **13 — Help** | Getting Started, Plot Types, Analysis Tools, Supported Formats, Tips |
| **14 — Settings** | Theme, DPI, plot defaults, named preset CRUD, plugin status |

## Supported File Formats (18)

| Category | Formats |
| --- | --- |
| Tabular | `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`, `.parquet` |
| Array | `.npy`, `.npz`, `.pkl` |
| Instrument / Spectral | `.jdx` / `.dx` (JCAMP-DX), `.h5` / `.hdf5` (HDF5), `.nc` / `.cdf` (NetCDF), `.spc` (Thermo Fisher), `.asc` (generic text), `.mzml` / `.mzxml` (mass spectrometry, optional) |

## CLI

Plottle includes a command-line interface with 5 subcommands:

```bash
python cli.py plot    <file> --plot <type> --x-column X --y-column Y [--output out.png]
python cli.py stats  <file>
python cli.py batch  <config.json> [--verbose]
python cli.py compare <file1> <file2> --plot line [--output comparison.png]
python cli.py convert <input> <output>
```

See the [CLI Tutorial](docs/tutorials/cli_guide.md) for full usage and examples.

## Python API

Plottle's modules can be used directly in scripts or Jupyter notebooks:

```python
import sys
sys.path.insert(0, 'path/to/plottle')

from modules.io import load_data
from modules.plotting import line_plot, save_figure
import numpy as np

wavelength = np.linspace(400, 800, 200)
absorbance = 0.8 * np.exp(-((wavelength - 520) ** 2) / (2 * 30 ** 2))

fig, ax, info = line_plot(
    wavelength,
    [absorbance],
    xlabel='Wavelength (nm)',
    ylabel='Absorbance',
    title='UV-Vis Spectrum',
    labels=['Sample A'],
)
save_figure(fig, 'spectrum.png', dpi=300)
```

### Key modules

| Module | Description |
| --- | --- |
| `modules.io` | `load_data()` / `save_data()` — auto-detects format from extension |
| `modules.math` | 25 functions — statistics, curve fitting, hypothesis tests, optimization, linear algebra |
| `modules.plotting` | 26 plot types; Matplotlib → `(fig, ax, info)`, Plotly → `(fig, info)` |
| `modules.signal` | 16 functions — smoothing, filtering, FFT, derivatives, baseline correction, interpolation |
| `modules.peaks` | 5 functions — find, integrate, FWHM, fit (Gaussian/Lorentzian/Voigt/pseudo-Voigt) |
| `modules.data_tools` | 12 non-destructive DataFrame operations |
| `modules.spectroscopy` | 18 functions — IR/Raman, UV-Vis, NMR, MS |
| `modules.nist` | NIST WebBook integration — fetch IR spectra by CAS number |
| `modules.batch` | Batch load, statistics, curve fit, peak analysis |
| `modules.annotations` | 7 overlay types (hline, vline, hspan, vspan, text, rectangle, ellipse) |
| `modules.report` | PDF report generation via `matplotlib.PdfPages` |
| `modules.molecular` | CPK atom data, Gaussian/ORCA/Molden parsers, Plotly 3D molecule builder |

## Plugin System

Drop a `plugin_*.py` file into `plugins/` to add custom plot types or analysis tools.
See [`plugins/plugin_example.py`](plugins/plugin_example.py) for the starter template.

## Dependencies

| Package | Purpose |
| --- | --- |
| `numpy` | Numerical computing |
| `scipy` | Scientific algorithms, curve fitting, signal processing |
| `pandas` | Tabular data |
| `matplotlib` | Static plotting |
| `seaborn` | Statistical visualizations |
| `plotly` | Interactive plots |
| `streamlit` | GUI framework |
| `openpyxl` | Excel file support |
| `requests` | NIST WebBook HTTP integration |
| `h5py` | HDF5 file support |
| `xarray` / `netcdf4` | NetCDF file support |

**Optional:**

| Package | Purpose |
| --- | --- |
| `statsmodels` | Two-way ANOVA (`anova_twoway` in `modules.math`) |
| `pymzml` | mzML/mzXML mass spectrometry files |
| `jcamp` | Alternative JCAMP-DX parser |

See [`requirements.txt`](requirements.txt) for pinned version ranges.

## Project Structure

```text
plottle/
├── modules/
│   ├── Home.py                     ← Streamlit entry point
│   ├── io.py                       ← 18-format data loader/saver
│   ├── math.py                     ← 25 analysis functions
│   ├── plotting.py                 ← 26 plot types (Matplotlib, Seaborn, Plotly)
│   ├── signal.py                   ← 16 signal processing functions
│   ├── peaks.py                    ← 5 peak analysis functions
│   ├── data_tools.py               ← 12 non-destructive DataFrame operations
│   ├── annotations.py              ← 7 Matplotlib overlay types
│   ├── spectroscopy.py             ← 18 spectroscopy functions
│   ├── nist.py                     ← NIST WebBook integration
│   ├── batch.py                    ← 5 batch processing functions
│   ├── report.py                   ← PDF report generation
│   ├── plugin_loader.py            ← plugin discovery and loading
│   ├── molecular/                  ← CPK atom data + vibrational parsers
│   ├── pages/                      ← 13 Streamlit pages (1_*.py – 14_*.py)
│   └── utils/                      ← session state, plot config, user settings
├── cli.py                          ← CLI entry point (5 subcommands)
├── tests/                          ← 17 test files, 900+ tests
├── plugins/                        ← plugin_example.py starter template
├── examples/                       ← 15+ standalone scripts + batch_config.json
├── notebooks/                      ← 4 Jupyter tutorials (Binder-ready)
├── docs/                           ← tutorials, gallery, cheatsheet
├── example-data/                   ← generated sample datasets
├── requirements.txt
├── pyproject.toml
└── launch.bat                      ← Windows one-click launcher
```

## Documentation

- [Getting Started](docs/getting_started.md) — installation walkthrough and API examples
- [Cheatsheet](docs/cheatsheet.md) — quick reference for all plot functions
- [CLI Tutorial](docs/tutorials/cli_guide.md) — step-by-step guide to all 5 CLI commands
- [GUI Tutorial](docs/tutorials/gui_guide.md) — page-by-page walkthrough of the Streamlit app
- [Examples](examples/) — 15+ standalone runnable scripts covering all major workflows

## About

**Author:** Jonathan D. Schultz, PhD — NCCU Department of Chemistry and Biochemistry
**Institution:** North Carolina Central University
**Use:** Research and teaching in computational science
**License:** See [LICENSE](LICENSE)
