"""Sphinx configuration for Plottle API documentation."""

import os
import sys

# Point Sphinx to the repo root so autodoc can import modules.*
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------

project = "Plottle"
copyright = "2026, Jonathan D. Schultz, PhD — NCCU Department of Chemistry and Biochemistry"
author = "Jonathan D. Schultz, PhD"
release = "2.0.0"
version = "2.0"

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # auto-generate docs from docstrings
    "sphinx.ext.napoleon",  # Google- and NumPy-style docstring support
    "sphinx.ext.viewcode",  # add [source] links next to each function
    "sphinx.ext.intersphinx",  # cross-reference NumPy / SciPy docs
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Napoleon settings — match the Google-style docstrings used in this project
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Mock imports that are not available in the Sphinx build environment or that
# should not be executed during doc generation (GUI-only dependencies).
autodoc_mock_imports = ["streamlit"]

# Exclude the Streamlit GUI entry point and page modules from autodoc.
# These files import streamlit at module level and are not part of the public API.
exclude_patterns = [
    "_build",
    "**/_build/*",
    "Thumbs.db",
    ".DS_Store",
]

templates_path = ["_templates"]

# -- HTML output ---------------------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 3,
    "titles_only": False,
    "collapse_navigation": False,
}

html_static_path = ["_static"]

html_title = "Plottle Documentation"
html_short_title = "Plottle"

# -- Source files --------------------------------------------------------------

source_suffix = ".rst"
master_doc = "index"
language = "en"
