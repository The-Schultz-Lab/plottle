"""Tests for modules/data_tools.py.

Coverage targets all 12 public functions and their key error paths.
"""

import math

import numpy as np
import pandas as pd
import pytest

from modules.data_tools import (
    add_formula_column,
    drop_nan,
    fill_nan,
    filter_rows,
    melt_dataframe,
    merge_dataframes,
    normalize_column,
    pivot_dataframe,
    resample_dataframe,
    rolling_transform,
    sort_dataframe,
    transpose_dataframe,
)


# ─── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_df():
    """A small numeric DataFrame for most tests."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "z": [10.0, 8.0, 6.0, 4.0, 2.0],
        }
    )


@pytest.fixture
def nan_df():
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [np.nan, 2.0, np.nan, 4.0, 5.0],
        }
    )


@pytest.fixture
def long_df():
    """Long-format DataFrame for pivot testing."""
    return pd.DataFrame(
        {
            "id":    ["r1", "r1", "r2", "r2"],
            "key":   ["A",  "B",  "A",  "B"],
            "value": [10,   20,   30,   40],
        }
    )


@pytest.fixture
def wide_df():
    """Wide-format DataFrame for melt testing."""
    return pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "A":      [10,   30],
            "B":      [20,   40],
        }
    )


# ─── add_formula_column ────────────────────────────────────────────────────────


class TestAddFormulaColumn:
    def test_simple_multiply(self, simple_df):
        result = add_formula_column(simple_df, "x2", "x * 2")
        assert "x2" in result.columns
        np.testing.assert_allclose(result["x2"].values, simple_df["x"].values * 2)

    def test_two_column_expression(self, simple_df):
        result = add_formula_column(simple_df, "ratio", "y / x")
        np.testing.assert_allclose(result["ratio"].values, np.full(5, 2.0))

    def test_math_function(self, simple_df):
        result = add_formula_column(simple_df, "lx", "log(x)")
        np.testing.assert_allclose(result["lx"].values, np.log(simple_df["x"].values))

    def test_mean_std_in_expression(self, simple_df):
        result = add_formula_column(simple_df, "norm_y", "(y - mean(y)) / std(y)")
        assert result["norm_y"].mean() == pytest.approx(0.0, abs=1e-10)
        assert result["norm_y"].std(ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_original_df_unchanged(self, simple_df):
        add_formula_column(simple_df, "tmp", "x + 1")
        assert "tmp" not in simple_df.columns

    def test_empty_expression_raises(self, simple_df):
        with pytest.raises(ValueError, match="expression must not be empty"):
            add_formula_column(simple_df, "col", "  ")

    def test_empty_new_col_raises(self, simple_df):
        with pytest.raises(ValueError, match="new_col must not be empty"):
            add_formula_column(simple_df, "  ", "x + 1")

    def test_bad_expression_raises(self, simple_df):
        with pytest.raises(Exception):
            add_formula_column(simple_df, "bad", "undefined_var + 1")

    def test_constants_available(self, simple_df):
        result = add_formula_column(simple_df, "pi_col", "x * pi")
        np.testing.assert_allclose(
            result["pi_col"].values, simple_df["x"].values * math.pi
        )


# ─── normalize_column ─────────────────────────────────────────────────────────


class TestNormalizeColumn:
    def test_minmax(self, simple_df):
        result = normalize_column(simple_df, "y", "min-max")
        col = result["y_norm"]
        assert col.min() == pytest.approx(0.0)
        assert col.max() == pytest.approx(1.0)

    def test_zscore(self, simple_df):
        result = normalize_column(simple_df, "y", "z-score")
        col = result["y_norm"]
        assert col.mean() == pytest.approx(0.0, abs=1e-10)
        assert col.std(ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_pctmax(self, simple_df):
        result = normalize_column(simple_df, "y", "pct-max")
        assert result["y_norm"].max() == pytest.approx(1.0)

    def test_area(self, simple_df):
        result = normalize_column(simple_df, "y", "area")
        col = result["y_norm"].to_numpy()
        assert np.trapezoid(np.abs(col)) == pytest.approx(1.0, rel=1e-6)

    def test_custom_output_column(self, simple_df):
        result = normalize_column(simple_df, "y", "min-max", new_col="y_scaled")
        assert "y_scaled" in result.columns

    def test_original_unchanged(self, simple_df):
        normalize_column(simple_df, "y", "z-score")
        assert "y_norm" not in simple_df.columns

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(KeyError):
            normalize_column(simple_df, "nonexistent", "min-max")

    def test_bad_method_raises(self, simple_df):
        with pytest.raises(ValueError, match="method must be one of"):
            normalize_column(simple_df, "y", "invalid")

    def test_constant_column_minmax_raises(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
        with pytest.raises(ValueError, match="constant"):
            normalize_column(df, "a", "min-max")

    def test_zero_variance_zscore_raises(self):
        df = pd.DataFrame({"a": [3.0, 3.0, 3.0]})
        with pytest.raises(ValueError, match="zero variance"):
            normalize_column(df, "a", "z-score")


# ─── transpose_dataframe ──────────────────────────────────────────────────────


class TestTransposeDataframe:
    def test_shape_swapped(self, simple_df):
        result = transpose_dataframe(simple_df)
        # original: 5 rows × 3 cols → transposed: 3 rows + 1 index col = 3 rows × (5+1) cols
        assert result.shape[0] == len(simple_df.columns)

    def test_returns_copy(self, simple_df):
        result = transpose_dataframe(simple_df)
        assert result is not simple_df

    def test_column_index_is_string(self, simple_df):
        result = transpose_dataframe(simple_df)
        # All column labels should be strings
        assert all(isinstance(c, str) for c in result.columns)

    def test_index_column_added(self, simple_df):
        result = transpose_dataframe(simple_df)
        assert "column" in result.columns


# ─── pivot_dataframe ──────────────────────────────────────────────────────────


class TestPivotDataframe:
    def test_basic_pivot(self, long_df):
        result = pivot_dataframe(long_df, "id", "key", "value")
        assert "A" in result.columns and "B" in result.columns
        assert result.shape == (2, 3)  # id + A + B

    def test_missing_column_raises(self, long_df):
        with pytest.raises(KeyError):
            pivot_dataframe(long_df, "id", "missing", "value")

    def test_returns_new_df(self, long_df):
        result = pivot_dataframe(long_df, "id", "key", "value")
        assert result is not long_df


# ─── melt_dataframe ───────────────────────────────────────────────────────────


class TestMeltDataframe:
    def test_basic_melt(self, wide_df):
        result = melt_dataframe(wide_df, id_vars=["sample"])
        assert "variable" in result.columns and "value" in result.columns
        assert len(result) == 4  # 2 rows × 2 value cols

    def test_custom_var_value_names(self, wide_df):
        result = melt_dataframe(
            wide_df, id_vars=["sample"], var_name="metric", value_name="measurement"
        )
        assert "metric" in result.columns and "measurement" in result.columns

    def test_explicit_value_vars(self, wide_df):
        result = melt_dataframe(wide_df, id_vars=["sample"], value_vars=["A"])
        assert len(result) == 2  # only column A melted

    def test_returns_new_df(self, wide_df):
        result = melt_dataframe(wide_df, id_vars=["sample"])
        assert result is not wide_df


# ─── filter_rows ──────────────────────────────────────────────────────────────


class TestFilterRows:
    def test_keep_matching(self, simple_df):
        result = filter_rows(simple_df, "x > 2")
        assert len(result) == 3
        assert result["x"].min() > 2

    def test_drop_matching(self, simple_df):
        result = filter_rows(simple_df, "x > 2", keep=False)
        assert len(result) == 2

    def test_empty_condition_raises(self, simple_df):
        with pytest.raises(ValueError, match="condition must not be empty"):
            filter_rows(simple_df, "  ")

    def test_index_reset(self, simple_df):
        result = filter_rows(simple_df, "x > 2")
        assert list(result.index) == list(range(len(result)))

    def test_original_unchanged(self, simple_df):
        filter_rows(simple_df, "x > 2")
        assert len(simple_df) == 5


# ─── sort_dataframe ───────────────────────────────────────────────────────────


class TestSortDataframe:
    def test_ascending(self, simple_df):
        shuffled = simple_df.sample(frac=1, random_state=42)
        result = sort_dataframe(shuffled, ["x"], ascending=True)
        assert list(result["x"]) == sorted(result["x"])

    def test_descending(self, simple_df):
        result = sort_dataframe(simple_df, ["x"], ascending=False)
        assert list(result["x"]) == sorted(result["x"], reverse=True)

    def test_multi_column_sort(self, simple_df):
        result = sort_dataframe(simple_df, ["y", "x"])
        assert list(result["y"]) == sorted(result["y"])

    def test_empty_by_raises(self, simple_df):
        with pytest.raises(ValueError, match="at least one column"):
            sort_dataframe(simple_df, [])

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(KeyError):
            sort_dataframe(simple_df, ["nonexistent"])

    def test_index_reset(self, simple_df):
        result = sort_dataframe(simple_df, ["z"], ascending=False)
        assert list(result.index) == list(range(len(result)))


# ─── merge_dataframes ─────────────────────────────────────────────────────────


class TestMergeDataframes:
    @pytest.fixture
    def left(self):
        return pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})

    @pytest.fixture
    def right(self):
        return pd.DataFrame({"id": [2, 3, 4], "b": [200, 300, 400]})

    def test_inner_join(self, left, right):
        result = merge_dataframes(left, right, on="id", how="inner")
        assert len(result) == 2
        assert set(result["id"]) == {2, 3}

    def test_left_join(self, left, right):
        result = merge_dataframes(left, right, on="id", how="left")
        assert len(result) == 3
        assert result["b"].isna().sum() == 1  # id=1 has no match

    def test_outer_join(self, left, right):
        result = merge_dataframes(left, right, on="id", how="outer")
        assert len(result) == 4

    def test_invalid_how_raises(self, left, right):
        with pytest.raises(ValueError, match="how must be one of"):
            merge_dataframes(left, right, on="id", how="cross")

    def test_missing_key_in_left_raises(self, left, right):
        with pytest.raises(KeyError, match="left DataFrame"):
            merge_dataframes(left, right, on="missing", how="inner")

    def test_missing_key_in_right_raises(self, left, right):
        with pytest.raises(KeyError, match="right DataFrame"):
            merge_dataframes(left, right, on="a", how="inner")


# ─── fill_nan ─────────────────────────────────────────────────────────────────


class TestFillNan:
    def test_fill_mean(self, nan_df):
        result = fill_nan(nan_df, method="mean")
        assert result.isna().sum().sum() == 0
        # Mean of [1,3,5] = 3.0
        assert result["a"].iloc[1] == pytest.approx(3.0)

    def test_fill_median(self, nan_df):
        result = fill_nan(nan_df, method="median")
        assert result.isna().sum().sum() == 0

    def test_fill_zero(self, nan_df):
        result = fill_nan(nan_df, method="zero")
        assert result.isna().sum().sum() == 0
        assert result["a"].iloc[1] == 0.0

    def test_fill_interpolate(self, nan_df):
        result = fill_nan(nan_df, method="interpolate")
        assert result["a"].isna().sum() == 0

    def test_fill_forward(self, nan_df):
        result = fill_nan(nan_df, method="forward")
        # After forward-fill, [1, nan, 3, nan, 5] → [1, 1, 3, 3, 5]
        assert result["a"].iloc[1] == pytest.approx(1.0)

    def test_fill_backward(self, nan_df):
        result = fill_nan(nan_df, method="backward")
        assert result["a"].iloc[1] == pytest.approx(3.0)

    def test_specific_columns(self, nan_df):
        result = fill_nan(nan_df, columns=["a"], method="zero")
        assert result["a"].isna().sum() == 0
        # Column b should be unchanged
        assert result["b"].isna().sum() == nan_df["b"].isna().sum()

    def test_bad_method_raises(self, nan_df):
        with pytest.raises(ValueError, match="method must be one of"):
            fill_nan(nan_df, method="invalid")

    def test_missing_column_raises(self, nan_df):
        with pytest.raises(KeyError):
            fill_nan(nan_df, columns=["nonexistent"])

    def test_original_unchanged(self, nan_df):
        fill_nan(nan_df)
        assert nan_df.isna().sum().sum() > 0


# ─── drop_nan ─────────────────────────────────────────────────────────────────


class TestDropNan:
    def test_drop_all_nan_rows(self, nan_df):
        result = drop_nan(nan_df)
        assert result.isna().sum().sum() == 0
        # Only row 4 (index 4, value 5.0 / 5.0) has no NaN
        assert len(result) == 1

    def test_drop_subset(self, nan_df):
        # Only drop rows with NaN in column 'a'
        result = drop_nan(nan_df, columns=["a"])
        assert result["a"].isna().sum() == 0
        assert len(result) == 3  # rows at original idx 0, 2, 4

    def test_index_reset(self, nan_df):
        result = drop_nan(nan_df)
        assert list(result.index) == list(range(len(result)))

    def test_original_unchanged(self, nan_df):
        drop_nan(nan_df)
        assert len(nan_df) == 5


# ─── resample_dataframe ───────────────────────────────────────────────────────


class TestResampleDataframe:
    def test_basic_resample(self, simple_df):
        result = resample_dataframe(simple_df, "x", ["y"], n_points=10)
        assert len(result) == 10
        assert "x" in result.columns and "y" in result.columns

    def test_linear_interp(self, simple_df):
        result = resample_dataframe(simple_df, "x", ["y"], n_points=9, method="linear")
        # y = 2x, so y at x=3 should be 6
        mid_idx = 4  # middle of 9 points from 1 to 5 → x ≈ 3
        np.testing.assert_allclose(result["y"].iloc[mid_idx], 6.0, atol=0.1)

    def test_custom_x_range(self, simple_df):
        result = resample_dataframe(
            simple_df, "x", ["y"], n_points=5, x_min=2.0, x_max=4.0
        )
        assert result["x"].min() == pytest.approx(2.0)
        assert result["x"].max() == pytest.approx(4.0)

    def test_missing_x_col_raises(self, simple_df):
        with pytest.raises(KeyError):
            resample_dataframe(simple_df, "missing", ["y"], n_points=5)

    def test_missing_y_col_raises(self, simple_df):
        with pytest.raises(KeyError):
            resample_dataframe(simple_df, "x", ["missing"], n_points=5)

    def test_bad_method_raises(self, simple_df):
        with pytest.raises(ValueError, match="method must be one of"):
            resample_dataframe(simple_df, "x", ["y"], n_points=5, method="spline")

    def test_n_points_too_small_raises(self, simple_df):
        with pytest.raises(ValueError, match="n_points"):
            resample_dataframe(simple_df, "x", ["y"], n_points=1)

    def test_non_monotonic_x_raises(self):
        df = pd.DataFrame({"x": [1, 3, 2, 4, 5], "y": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="monotonic"):
            resample_dataframe(df, "x", ["y"], n_points=10)


# ─── rolling_transform ────────────────────────────────────────────────────────


class TestRollingTransform:
    def test_rolling_mean(self, simple_df):
        result = rolling_transform(simple_df, ["y"], operation="rolling_mean", window=3)
        assert "y_rolling_mean" in result.columns
        assert len(result) == len(simple_df)

    def test_rolling_sum(self, simple_df):
        result = rolling_transform(simple_df, ["y"], operation="rolling_sum", window=3)
        assert "y_rolling_sum" in result.columns

    def test_cumsum(self, simple_df):
        result = rolling_transform(simple_df, ["y"], operation="cumsum")
        expected_last = simple_df["y"].sum()
        assert result["y_cumsum"].iloc[-1] == pytest.approx(expected_last)

    def test_cumprod(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = rolling_transform(df, ["a"], operation="cumprod")
        assert result["a_cumprod"].iloc[-1] == pytest.approx(24.0)

    def test_multiple_columns(self, simple_df):
        result = rolling_transform(simple_df, ["y", "z"], operation="cumsum")
        assert "y_cumsum" in result.columns
        assert "z_cumsum" in result.columns

    def test_bad_operation_raises(self, simple_df):
        with pytest.raises(ValueError, match="operation must be one of"):
            rolling_transform(simple_df, ["y"], operation="invalid")

    def test_bad_window_raises(self, simple_df):
        with pytest.raises(ValueError, match="window must be >= 1"):
            rolling_transform(simple_df, ["y"], window=0)

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(KeyError):
            rolling_transform(simple_df, ["nonexistent"])

    def test_original_unchanged(self, simple_df):
        rolling_transform(simple_df, ["y"], operation="cumsum")
        assert "y_cumsum" not in simple_df.columns
