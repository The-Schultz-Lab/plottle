"""Session State Management for Streamlit GUI.

This module provides centralized session state management to persist data,
plots, and analysis results across page navigation in the Streamlit multi-page app.

Functions
---------
initialize_session_state()
    Initialize all session state variables with defaults
add_dataset(name, data, metadata=None)
    Add a dataset to session state
get_current_dataset()
    Retrieve the currently selected dataset
get_dataset(name)
    Retrieve a specific dataset by name
delete_dataset(name)
    Remove a dataset from session state
add_plot_to_history(plot_data)
    Add a generated plot to history
clear_plot_history()
    Remove all plots from history
add_analysis_result(result_data)
    Add analysis result to session state
save_session_to_file(filepath)
    Persist session state to JSON file
load_session_from_file(filepath)
    Restore session state from JSON file
clear_session()
    Reset all session state to defaults

Examples
--------
>>> import streamlit as st
>>> from modules.utils.session_state import initialize_session_state, add_dataset
>>> initialize_session_state()
>>> data = np.array([1, 2, 3, 4, 5])
>>> add_dataset('test_data.npy', data)
>>> dataset = get_current_dataset()
"""

import streamlit as st
import json
import pickle
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def initialize_session_state():
    """Initialize all session state variables with default values.

    This should be called once at the start of the app (in gui.py).
    Creates the following state variables:
    - datasets: Dict[str, Any] - Loaded datasets keyed by filename
    - current_dataset: str | None - Name of currently selected dataset
    - plot_history: List[Dict] - History of generated plots
    - analysis_results: List[Dict] - Results from mathematical operations
    - plot_config: Dict - Current plot configuration settings
    - export_queue: List[Dict] - Items queued for export
    """
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}

    if "dataset_metadata" not in st.session_state:
        st.session_state.dataset_metadata = {}

    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None

    if "plot_history" not in st.session_state:
        st.session_state.plot_history = []

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []

    if "plot_config" not in st.session_state:
        st.session_state.plot_config = {
            "default_figsize": (8, 6),
            "default_dpi": 100,
            "default_style": "default",
        }

    if "export_queue" not in st.session_state:
        st.session_state.export_queue = []

    if "mol_vib_data" not in st.session_state:
        st.session_state.mol_vib_data = None


def add_dataset(name: str, data: Any, metadata: Optional[Dict] = None):
    """Add a dataset to session state.

    Parameters
    ----------
    name : str
        Dataset name (usually filename)
    data : Any
        Dataset (DataFrame, ndarray, dict, etc.)
    metadata : dict, optional
        Additional metadata (file size, upload time, etc.)
    """
    st.session_state.datasets[name] = data

    # Store metadata
    if metadata is None:
        metadata = {}

    metadata["added_time"] = datetime.now().isoformat()
    metadata["data_type"] = type(data).__name__

    # Add shape/size info
    if isinstance(data, pd.DataFrame):
        metadata["shape"] = data.shape
        metadata["columns"] = list(data.columns)
        metadata["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}
    elif isinstance(data, np.ndarray):
        metadata["shape"] = data.shape
        metadata["dtype"] = str(data.dtype)
    elif isinstance(data, (list, tuple)):
        metadata["length"] = len(data)

    st.session_state.dataset_metadata[name] = metadata

    # Set as current dataset if it's the first one
    if st.session_state.current_dataset is None:
        st.session_state.current_dataset = name


def get_current_dataset() -> Optional[Any]:
    """Retrieve the currently selected dataset.

    Returns
    -------
    data : Any or None
        Current dataset, or None if no dataset is selected
    """
    if st.session_state.current_dataset is None:
        return None

    return st.session_state.datasets.get(st.session_state.current_dataset)


def get_dataset(name: str) -> Optional[Any]:
    """Retrieve a specific dataset by name.

    Parameters
    ----------
    name : str
        Dataset name

    Returns
    -------
    data : Any or None
        Dataset if found, None otherwise
    """
    return st.session_state.datasets.get(name)


def delete_dataset(name: str) -> bool:
    """Remove a dataset from session state.

    Parameters
    ----------
    name : str
        Dataset name to delete

    Returns
    -------
    success : bool
        True if dataset was deleted, False if not found
    """
    if name in st.session_state.datasets:
        del st.session_state.datasets[name]
        del st.session_state.dataset_metadata[name]

        # Update current dataset if we deleted it
        if st.session_state.current_dataset == name:
            if st.session_state.datasets:
                st.session_state.current_dataset = list(st.session_state.datasets.keys())[0]
            else:
                st.session_state.current_dataset = None

        return True

    return False


_MAX_PLOT_HISTORY = 50


def add_plot_to_history(plot_data: Dict):
    """Add a generated plot to history (capped at ``_MAX_PLOT_HISTORY`` entries).

    Parameters
    ----------
    plot_data : dict
        Plot information with keys:
        - type: Plot type (e.g., 'histogram', 'scatter')
        - dataset: Dataset name used
        - config: Plot configuration dict
        - timestamp: When plot was created
        - figure: Matplotlib/Plotly figure object (optional)

    Notes
    -----
    Figure objects are stripped from older entries once the history exceeds
    five entries to prevent unbounded memory growth.
    """
    if "timestamp" not in plot_data:
        plot_data["timestamp"] = datetime.now().isoformat()

    st.session_state.plot_history.append(plot_data)

    # Drop figure objects from all but the 5 most recent entries.
    history = st.session_state.plot_history
    for entry in history[:-5]:
        entry.pop("figure", None)

    # Rolling cap — evict oldest entries when limit is exceeded.
    if len(history) > _MAX_PLOT_HISTORY:
        st.session_state.plot_history = history[-_MAX_PLOT_HISTORY:]


def clear_plot_history():
    """Remove all plots from history."""
    st.session_state.plot_history = []


def add_analysis_result(result_data: Dict):
    """Add analysis result to session state.

    Parameters
    ----------
    result_data : dict
        Analysis result with keys:
        - type: Analysis type (e.g., 'statistics', 'curve_fit')
        - dataset: Dataset name used
        - results: Analysis output dict
        - timestamp: When analysis was performed
    """
    if "timestamp" not in result_data:
        result_data["timestamp"] = datetime.now().isoformat()

    st.session_state.analysis_results.append(result_data)


def _serialize_data(data: Any) -> Dict:
    """Serialize data for JSON storage.

    Parameters
    ----------
    data : Any
        Data to serialize

    Returns
    -------
    serialized : dict
        Serialized representation
    """
    if isinstance(data, pd.DataFrame):
        return {"__type__": "DataFrame", "__data__": data.to_json(orient="split")}
    elif isinstance(data, np.ndarray):
        return {
            "__type__": "ndarray",
            "__data__": base64.b64encode(pickle.dumps(data)).decode("utf-8"),
        }
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    elif isinstance(data, (list, tuple)):
        return [_serialize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: _serialize_data(value) for key, value in data.items()}
    else:
        # Fallback: pickle and base64 encode
        return {
            "__type__": "pickled",
            "__data__": base64.b64encode(pickle.dumps(data)).decode("utf-8"),
        }


def _deserialize_data(data: Any) -> Any:
    """Deserialize data from JSON storage.

    Parameters
    ----------
    data : Any
        Serialized data

    Returns
    -------
    deserialized : Any
        Deserialized data
    """
    if isinstance(data, dict):
        if "__type__" in data:
            if data["__type__"] == "DataFrame":
                return pd.read_json(data["__data__"], orient="split")
            elif data["__type__"] == "ndarray":
                return pickle.loads(base64.b64decode(data["__data__"]))
            elif data["__type__"] == "pickled":
                return pickle.loads(base64.b64decode(data["__data__"]))
        else:
            return {key: _deserialize_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_deserialize_data(item) for item in data]
    else:
        return data


def save_session_to_file(filepath: str):
    """Persist session state to JSON file.

    Parameters
    ----------
    filepath : str
        Path to save session file

    Notes
    -----
    This saves datasets, plot history, and analysis results.
    Matplotlib figure objects are NOT saved (too large).
    """
    session_data = {
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "datasets": {},
        "dataset_metadata": st.session_state.dataset_metadata,
        "current_dataset": st.session_state.current_dataset,
        "plot_history": [
            {k: v for k, v in plot.items() if k != "figure"}
            for plot in st.session_state.plot_history
        ],
        "analysis_results": st.session_state.analysis_results,
        "plot_config": st.session_state.plot_config,
    }

    # Serialize datasets
    for name, data in st.session_state.datasets.items():
        session_data["datasets"][name] = _serialize_data(data)

    filepath = Path(filepath)
    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)


def load_session_from_file(filepath: str):
    """Restore session state from JSON file.

    Parameters
    ----------
    filepath : str
        Path to session file
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        session_data = json.load(f)

    # Deserialize datasets
    st.session_state.datasets = {}
    for name, data in session_data["datasets"].items():
        st.session_state.datasets[name] = _deserialize_data(data)

    st.session_state.dataset_metadata = session_data["dataset_metadata"]
    st.session_state.current_dataset = session_data["current_dataset"]
    st.session_state.plot_history = session_data["plot_history"]
    st.session_state.analysis_results = session_data["analysis_results"]
    st.session_state.plot_config = session_data["plot_config"]


def clear_session():
    """Reset all session state to defaults."""
    st.session_state.datasets = {}
    st.session_state.dataset_metadata = {}
    st.session_state.current_dataset = None
    st.session_state.plot_history = []
    st.session_state.analysis_results = []
    st.session_state.plot_config = {
        "default_figsize": (8, 6),
        "default_dpi": 100,
        "default_style": "default",
    }
    st.session_state.export_queue = []


def get_session_summary() -> Dict:
    """Get a summary of current session state.

    Returns
    -------
    summary : dict
        Summary with counts and current selections
    """
    return {
        "num_datasets": len(st.session_state.datasets),
        "current_dataset": st.session_state.current_dataset,
        "num_plots": len(st.session_state.plot_history),
        "num_analyses": len(st.session_state.analysis_results),
        "dataset_names": list(st.session_state.datasets.keys()),
    }
