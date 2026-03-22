"""Data Manipulation Module.

Non-destructive DataFrame transformation operations.  Every function accepts a
``pandas.DataFrame`` and returns a **new** DataFrame; the original is never
modified in-place.

Functions
---------
Column transforms:
    add_formula_column     -- compute a new column from a safe expression
    normalize_column       -- min-max, z-score, percent-of-max, area

Shape / structure:
    transpose_dataframe    -- rows ↔ columns
    pivot_dataframe        -- long → wide (pivot)
    melt_dataframe         -- wide → long (melt)

Filtering / sorting:
    filter_rows            -- keep or drop rows matching a condition string
    sort_dataframe         -- sort by one or more columns

Merging:
    merge_dataframes       -- join two DataFrames on a key column

Missing values:
    fill_nan               -- fill NaN with a strategy (mean/median/zero/interpolate)
    drop_nan               -- drop rows that contain NaN

Time-series transforms:
    resample_dataframe     -- upsample or downsample to a new x-grid (cubic/linear/nearest)
    rolling_transform      -- rolling mean / rolling sum / cumulative sum per column
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ─── Column transforms ────────────────────────────────────────────────────────


# Safe built-in namespace made available inside formula expressions.
_SAFE_MATH = {
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "pi": math.pi,
    "e": math.e,
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "cumsum": np.cumsum,
    "diff": np.diff,
    "nan": np.nan,
}


def add_formula_column(
    df: pd.DataFrame,
    new_col: str,
    expression: str,
) -> pd.DataFrame:
    """Compute a new column from a Python/NumPy expression.

    Column names in the DataFrame are injected into the evaluation namespace
    as local variables so they can be referenced directly in the expression.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not modified).
    new_col : str
        Name for the new column.
    expression : str
        A Python expression using column names and the math functions listed
        below.  For example: ``"log(A) / B"`` or ``"(A - mean(A)) / std(A)"``.

        Available functions: ``log``, ``log2``, ``log10``, ``exp``, ``sqrt``,
        ``abs``, ``sin``, ``cos``, ``tan``, ``mean``, ``std``, ``min``,
        ``max``, ``sum``, ``cumsum``, ``diff`` (all operate on arrays).
        Constants: ``pi``, ``e``, ``nan``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *new_col* appended.

    Raises
    ------
    ValueError
        If *expression* is empty or *new_col* is empty.
    Exception
        Re-raised if the expression itself raises (e.g. ZeroDivisionError,
        NameError for unknown column names).
    """
    if not expression.strip():
        raise ValueError("expression must not be empty")
    if not new_col.strip():
        raise ValueError("new_col must not be empty")

    # Build evaluation namespace: math helpers + column arrays
    namespace = dict(_SAFE_MATH)
    for col in df.columns:
        namespace[col] = df[col].to_numpy(dtype=float, na_value=np.nan)

    result = eval(expression, {"__builtins__": {}}, namespace)  # noqa: S307

    out = df.copy()
    out[new_col] = result
    return out


def normalize_column(
    df: pd.DataFrame,
    column: str,
    method: str = "min-max",
    new_col: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a numeric column and store the result.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not modified).
    column : str
        Name of the column to normalize.
    method : str
        Normalization strategy.  One of:

        - ``'min-max'``   — scale to [0, 1]
        - ``'z-score'``   — subtract mean, divide by std
        - ``'pct-max'``   — divide by maximum (percentage of max)
        - ``'area'``      — divide by the trapezoidal area (unit-area signal)
    new_col : str, optional
        Name for the output column.  Defaults to ``"<column>_norm"``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the normalized column appended (or replaced if
        *new_col* already exists).

    Raises
    ------
    KeyError
        If *column* is not in *df*.
    ValueError
        If *method* is not one of the supported options, or the column
        has no variance / area == 0 (would cause division by zero).
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    methods = ("min-max", "z-score", "pct-max", "area")
    if method not in methods:
        raise ValueError(f"method must be one of {methods}; got '{method}'")

    y = df[column].to_numpy(dtype=float)

    if method == "min-max":
        lo, hi = np.nanmin(y), np.nanmax(y)
        if hi == lo:
            raise ValueError(f"Column '{column}' is constant — cannot apply min-max normalization")
        result = (y - lo) / (hi - lo)

    elif method == "z-score":
        mu = np.nanmean(y)
        sigma = np.nanstd(y)
        if sigma == 0:
            raise ValueError(
                f"Column '{column}' has zero variance — cannot apply z-score normalization"
            )
        result = (y - mu) / sigma

    elif method == "pct-max":
        mx = np.nanmax(np.abs(y))
        if mx == 0:
            raise ValueError(f"Column '{column}' is all-zero — cannot apply pct-max normalization")
        result = y / mx

    else:  # area
        area = np.trapezoid(np.abs(y))
        if area == 0:
            raise ValueError(f"Column '{column}' has zero area — cannot apply area normalization")
        result = y / area

    out = df.copy()
    col_name = new_col if new_col else f"{column}_norm"
    out[col_name] = result
    return out


# ─── Shape / structure ────────────────────────────────────────────────────────


def transpose_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Transpose a DataFrame (rows ↔ columns).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Transposed copy with reset index and columns converted to strings.
    """
    transposed = df.T.copy()
    # Convert column index (originally row index) to strings for consistency
    transposed.columns = [str(c) for c in transposed.columns]
    transposed = transposed.reset_index()
    transposed = transposed.rename(columns={"index": "column"})
    return transposed


def pivot_dataframe(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
) -> pd.DataFrame:
    """Pivot a long-format DataFrame to wide format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame in long format.
    index : str
        Column to use as the new row index.
    columns : str
        Column whose unique values become new column headers.
    values : str
        Column to use for cell values.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with reset index.

    Raises
    ------
    KeyError
        If any of *index*, *columns*, or *values* are not in *df*.
    """
    for col in (index, columns, values):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    result = df.pivot(index=index, columns=columns, values=values)
    result.columns.name = None
    result = result.reset_index()
    return result


def melt_dataframe(
    df: pd.DataFrame,
    id_vars: List[str],
    value_vars: Optional[List[str]] = None,
    var_name: str = "variable",
    value_name: str = "value",
) -> pd.DataFrame:
    """Melt a wide-format DataFrame to long format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame in wide format.
    id_vars : list of str
        Columns to keep as identifier variables.
    value_vars : list of str, optional
        Columns to melt.  Defaults to all columns not in *id_vars*.
    var_name : str
        Name for the variable column in the output.
    value_name : str
        Name for the value column in the output.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame.
    """
    return df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )


# ─── Filtering / sorting ──────────────────────────────────────────────────────


def filter_rows(
    df: pd.DataFrame,
    condition: str,
    keep: bool = True,
) -> pd.DataFrame:
    """Filter rows using a pandas ``query``-style condition string.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not modified).
    condition : str
        A pandas ``DataFrame.query`` expression such as ``"A > 0"`` or
        ``"B == 'control'"`` or ``"A > 0 and C < 10"``.
    keep : bool
        If ``True`` (default), keep rows that match the condition.
        If ``False``, drop rows that match (i.e. keep the complement).

    Returns
    -------
    pd.DataFrame
        Filtered copy with reset index.

    Raises
    ------
    ValueError
        If *condition* is empty.
    Exception
        Re-raised if ``query`` fails (bad syntax, unknown column, etc.).
    """
    if not condition.strip():
        raise ValueError("condition must not be empty")

    mask = df.eval(condition)
    if not keep:
        mask = ~mask
    return df[mask].reset_index(drop=True)


def sort_dataframe(
    df: pd.DataFrame,
    by: List[str],
    ascending: bool = True,
) -> pd.DataFrame:
    """Sort a DataFrame by one or more columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not modified).
    by : list of str
        Column name(s) to sort by.
    ascending : bool
        Sort direction.  Use ``True`` for ascending (default), ``False``
        for descending.

    Returns
    -------
    pd.DataFrame
        Sorted copy with reset index.

    Raises
    ------
    KeyError
        If any column in *by* is not in *df*.
    ValueError
        If *by* is empty.
    """
    if not by:
        raise ValueError("'by' must contain at least one column name")
    for col in by:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    return df.sort_values(by=by, ascending=ascending).reset_index(drop=True)


# ─── Merging ──────────────────────────────────────────────────────────────────


def merge_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    how: str = "inner",
    suffixes: tuple = ("_left", "_right"),
) -> pd.DataFrame:
    """Join two DataFrames on a shared key column.

    Parameters
    ----------
    left : pd.DataFrame
        Left DataFrame.
    right : pd.DataFrame
        Right DataFrame.
    on : str
        Column name to join on (must exist in both DataFrames).
    how : str
        Type of join: ``'inner'``, ``'left'``, ``'right'``, or ``'outer'``.
    suffixes : tuple of str
        Suffixes to append to overlapping column names.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.

    Raises
    ------
    KeyError
        If *on* is not present in both DataFrames.
    ValueError
        If *how* is not a valid join type.
    """
    valid_how = ("inner", "left", "right", "outer")
    if how not in valid_how:
        raise ValueError(f"how must be one of {valid_how}; got '{how}'")
    if on not in left.columns:
        raise KeyError(f"Key column '{on}' not found in left DataFrame")
    if on not in right.columns:
        raise KeyError(f"Key column '{on}' not found in right DataFrame")

    return pd.merge(left, right, on=on, how=how, suffixes=suffixes)


# ─── Missing values ───────────────────────────────────────────────────────────


def fill_nan(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "mean",
) -> pd.DataFrame:
    """Fill NaN values in selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not modified).
    columns : list of str, optional
        Columns to fill.  Defaults to all numeric columns.
    method : str
        Fill strategy.  One of:

        - ``'mean'``          — replace with column mean
        - ``'median'``        — replace with column median
        - ``'zero'``          — replace with 0
        - ``'interpolate'``   — linear interpolation along the column
        - ``'forward'``       — forward-fill (propagate last valid value)
        - ``'backward'``      — backward-fill

    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaN values filled.

    Raises
    ------
    ValueError
        If *method* is not recognized.
    KeyError
        If any column in *columns* is not in *df*.
    """
    methods = ("mean", "median", "zero", "interpolate", "forward", "backward")
    if method not in methods:
        raise ValueError(f"method must be one of {methods}; got '{method}'")

    out = df.copy()
    cols = columns if columns is not None else list(df.select_dtypes(include="number").columns)

    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        if method == "mean":
            out[col] = out[col].fillna(out[col].mean())
        elif method == "median":
            out[col] = out[col].fillna(out[col].median())
        elif method == "zero":
            out[col] = out[col].fillna(0)
        elif method == "interpolate":
            out[col] = out[col].interpolate(method="linear", limit_direction="both")
        elif method == "forward":
            out[col] = out[col].ffill()
        elif method == "backward":
            out[col] = out[col].bfill()

    return out


def drop_nan(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Drop rows that contain NaN in any (or selected) columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not modified).
    columns : list of str, optional
        Only consider NaN in these columns.  If ``None``, any NaN in any
        column will trigger row removal.

    Returns
    -------
    pd.DataFrame
        Filtered copy with reset index.
    """
    return df.dropna(subset=columns).reset_index(drop=True)


# ─── Time-series transforms ───────────────────────────────────────────────────


def resample_dataframe(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    n_points: int,
    method: str = "cubic",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> pd.DataFrame:
    """Resample one or more y-columns to a new, uniform x-grid.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not modified).
    x_col : str
        Column to use as the independent variable (must be numeric, monotonic).
    y_cols : list of str
        Columns to resample.
    n_points : int
        Number of points in the new x-grid.
    method : str
        Interpolation method: ``'cubic'``, ``'linear'``, or ``'nearest'``.
    x_min : float, optional
        Lower bound of the new x-grid.  Defaults to ``min(x_col)``.
    x_max : float, optional
        Upper bound of the new x-grid.  Defaults to ``max(x_col)``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with *x_col* set to the uniform grid and each y-column
        resampled.

    Raises
    ------
    KeyError
        If *x_col* or any column in *y_cols* is not in *df*.
    ValueError
        If *method* is not supported, *n_points* < 2, or *x_col* is not
        monotonic.
    """
    if x_col not in df.columns:
        raise KeyError(f"Column '{x_col}' not found in DataFrame")
    for col in y_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    valid_methods = ("cubic", "linear", "nearest")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}; got '{method}'")

    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    x = df[x_col].to_numpy(dtype=float)
    if not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)):
        raise ValueError(f"Column '{x_col}' must be strictly monotonic for resampling")

    lo = x_min if x_min is not None else float(np.min(x))
    hi = x_max if x_max is not None else float(np.max(x))
    x_new = np.linspace(lo, hi, n_points)

    kind = "cubic" if method == "cubic" else method
    out = pd.DataFrame({x_col: x_new})
    for col in y_cols:
        y = df[col].to_numpy(dtype=float)
        f = interp1d(x, y, kind=kind, bounds_error=False, fill_value="extrapolate")
        out[col] = f(x_new)

    return out


def rolling_transform(
    df: pd.DataFrame,
    columns: List[str],
    operation: str = "rolling_mean",
    window: int = 5,
) -> pd.DataFrame:
    """Apply rolling or cumulative transforms to selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not modified).
    columns : list of str
        Columns to transform.
    operation : str
        Transform to apply.  One of:

        - ``'rolling_mean'``   — rolling (sliding-window) mean
        - ``'rolling_sum'``    — rolling sum
        - ``'cumsum'``         — cumulative sum
        - ``'cumprod'``        — cumulative product
    window : int
        Window size (only used for rolling operations).  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the transformed columns appended as
        ``"<col>_<operation>"``.

    Raises
    ------
    ValueError
        If *operation* is not recognized or *window* < 1.
    KeyError
        If any column in *columns* is not in *df*.
    """
    valid_ops = ("rolling_mean", "rolling_sum", "cumsum", "cumprod")
    if operation not in valid_ops:
        raise ValueError(f"operation must be one of {valid_ops}; got '{operation}'")
    if window < 1:
        raise ValueError("window must be >= 1")

    out = df.copy()
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        new_col = f"{col}_{operation}"
        if operation == "rolling_mean":
            out[new_col] = df[col].rolling(window=window, center=True, min_periods=1).mean()
        elif operation == "rolling_sum":
            out[new_col] = df[col].rolling(window=window, center=True, min_periods=1).sum()
        elif operation == "cumsum":
            out[new_col] = df[col].cumsum()
        elif operation == "cumprod":
            out[new_col] = df[col].cumprod()

    return out
