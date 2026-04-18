# -*- coding: utf-8 -*-
"""Plottle — Scientific Data Visualization and Analysis Toolkit.

A unified Python toolkit for scientific data visualization and analysis,
developed at North Carolina Central University for research and teaching
in computational science.

Modules
-------
plottle.io : Data I/O — 18 formats (CSV, Excel, JCAMP, HDF5, NetCDF, mzML, …)
plottle.math : Statistics, curve fitting, hypothesis testing, optimization
plottle.plotting : 26 plot types across Matplotlib, Seaborn, and Plotly
plottle.signal : Smoothing, filtering, FFT, baseline correction, interpolation
plottle.peaks : Peak detection, integration, FWHM, single- and multi-peak fitting
plottle.data_tools : Non-destructive DataFrame transforms (normalize, pivot, merge, …)
plottle.annotations : Overlay annotations on Matplotlib figures
plottle.spectroscopy : IR/Raman, UV-Vis, NMR, and mass spectrometry tools
plottle.nist : NIST WebBook IR spectrum fetching by CAS number
plottle.batch : Directory scanning, batch statistics, curve fit, and peak analysis
plottle.report : PDF report generation via matplotlib PdfPages
plottle.molecular : 3D molecular structure and vibrational mode visualization
plottle.Home : 14-page Streamlit GUI entry point

Examples
--------
>>> from plottle.plotting import histogram
>>> from plottle.io import load_data
>>> data = load_data('experiment.csv')
>>> fig, ax, info = histogram(data['value'], bins=30)
"""

__version__ = "2.0.0"
__author__ = "Jonathan D. Schultz, PhD — North Carolina Central University"
__all__ = ["modules"]
