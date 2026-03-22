# -*- coding: utf-8 -*-
"""Plottle — Scientific Data Visualization and Analysis Toolkit.

A unified Python toolkit for scientific data visualization and analysis,
developed at North Carolina Central University for research and teaching
in computational science.

Modules
-------
modules.io : Data I/O — 18 formats (CSV, Excel, JCAMP, HDF5, NetCDF, mzML, …)
modules.math : Statistics, curve fitting, hypothesis testing, optimization
modules.plotting : 26 plot types across Matplotlib, Seaborn, and Plotly
modules.signal : Smoothing, filtering, FFT, baseline correction, interpolation
modules.peaks : Peak detection, integration, FWHM, single- and multi-peak fitting
modules.data_tools : Non-destructive DataFrame transforms (normalize, pivot, merge, …)
modules.annotations : Overlay annotations on Matplotlib figures
modules.spectroscopy : IR/Raman, UV-Vis, NMR, and mass spectrometry tools
modules.nist : NIST WebBook IR spectrum fetching by CAS number
modules.batch : Directory scanning, batch statistics, curve fit, and peak analysis
modules.report : PDF report generation via matplotlib PdfPages
modules.molecular : 3D molecular structure and vibrational mode visualization
modules.Home : 14-page Streamlit GUI entry point

Examples
--------
>>> from modules.plotting import histogram
>>> from modules.io import load_data
>>> data = load_data('experiment.csv')
>>> fig, ax, info = histogram(data['value'], bins=30)
"""

__version__ = "2.0.0"
__author__ = "Jonathan D. Schultz, PhD — North Carolina Central University"
__all__ = ["modules"]
