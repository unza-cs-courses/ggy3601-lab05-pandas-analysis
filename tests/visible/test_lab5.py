#!/usr/bin/env python3
"""
Visible Tests for Lab 5: pandas Analysis
GGY3601 - Introduction to Programming for Geologists

These tests verify student implementations of pandas operations.
Students can see and run these tests locally.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# Part 1: DataFrame Basics Tests
# =============================================================================

class TestDataFrameBasics:
    """Tests for lab5_dataframe_basics.py"""

    def test_create_sample_dataframe(self):
        """Test creating a DataFrame from a dictionary."""
        from lab5_dataframe_basics import create_sample_dataframe

        data = {
            'sample_id': ['GEO-001', 'GEO-002', 'GEO-003'],
            'rock_type': ['Granite', 'Basalt', 'Sandstone'],
            'grade': [2.5, 1.8, 3.2]
        }

        df = create_sample_dataframe(data)

        assert df is not None, "Function should return a DataFrame"
        assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
        assert len(df) == 3, "DataFrame should have 3 rows"
        assert list(df.columns) == ['sample_id', 'rock_type', 'grade']

    def test_read_drilling_data(self, drilling_data_path):
        """Test reading CSV file into DataFrame."""
        from lab5_dataframe_basics import read_drilling_data

        if not drilling_data_path.exists():
            pytest.skip("Drilling data file not found")

        df = read_drilling_data(str(drilling_data_path))

        assert df is not None, "Function should return a DataFrame"
        assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
        assert len(df) == 200, "Drilling data should have 200 rows"
        assert 'sample_id' in df.columns, "DataFrame should have sample_id column"

    def test_get_dataframe_info(self, sample_dataframe):
        """Test extracting DataFrame information."""
        from lab5_dataframe_basics import get_dataframe_info

        info = get_dataframe_info(sample_dataframe)

        assert info is not None, "Function should return info dict"
        assert 'num_rows' in info, "Info should contain num_rows"
        assert 'num_cols' in info, "Info should contain num_cols"
        assert info['num_rows'] == 5
        assert info['num_cols'] == 6

    def test_display_first_n_rows(self, sample_dataframe):
        """Test getting first n rows."""
        from lab5_dataframe_basics import display_first_n_rows

        result = display_first_n_rows(sample_dataframe, n=3)

        assert result is not None
        assert len(result) == 3
        assert result.iloc[0]['sample_id'] == 'GEO-001'

    def test_get_column_names(self, sample_dataframe):
        """Test getting column names."""
        from lab5_dataframe_basics import get_column_names

        columns = get_column_names(sample_dataframe)

        assert columns is not None
        assert isinstance(columns, list)
        assert 'sample_id' in columns
        assert 'rock_type' in columns

    def test_get_numeric_columns(self, sample_dataframe):
        """Test identifying numeric columns."""
        from lab5_dataframe_basics import get_numeric_columns

        numeric_cols = get_numeric_columns(sample_dataframe)

        assert numeric_cols is not None
        assert 'grade' in numeric_cols
        assert 'depth' in numeric_cols
        assert 'rock_type' not in numeric_cols  # String column

    def test_check_missing_values(self, dataframe_with_nulls):
        """Test checking for missing values."""
        from lab5_dataframe_basics import check_missing_values

        missing = check_missing_values(dataframe_with_nulls)

        assert missing is not None
        assert missing['grade'] == 2  # Two missing grade values
        assert missing['sample_id'] == 0  # No missing sample_ids


# =============================================================================
# Part 2: Data Selection Tests
# =============================================================================

class TestDataSelection:
    """Tests for lab5_selection.py"""

    def test_select_single_column(self, sample_dataframe):
        """Test selecting a single column."""
        from lab5_selection import select_single_column

        result = select_single_column(sample_dataframe, 'grade')

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == 5
        assert result.iloc[0] == 2.5

    def test_select_multiple_columns(self, sample_dataframe):
        """Test selecting multiple columns."""
        from lab5_selection import select_multiple_columns

        result = select_multiple_columns(sample_dataframe, ['sample_id', 'grade'])

        assert result is not None
        assert list(result.columns) == ['sample_id', 'grade']
        assert len(result) == 5

    def test_filter_by_value(self, sample_dataframe):
        """Test filtering by exact value."""
        from lab5_selection import filter_by_value

        result = filter_by_value(sample_dataframe, 'rock_type', 'Granite')

        assert result is not None
        assert len(result) == 2
        assert all(result['rock_type'] == 'Granite')

    def test_filter_greater_than(self, sample_dataframe):
        """Test filtering with greater than condition."""
        from lab5_selection import filter_greater_than

        result = filter_greater_than(sample_dataframe, 'grade', 2.0)

        assert result is not None
        assert len(result) == 3
        assert all(result['grade'] > 2.0)

    def test_filter_between(self, sample_dataframe):
        """Test filtering between values."""
        from lab5_selection import filter_between

        result = filter_between(sample_dataframe, 'depth', 100, 300)

        assert result is not None
        assert all((result['depth'] >= 100) & (result['depth'] <= 300))

    def test_filter_in_list(self, sample_dataframe):
        """Test filtering by list of values."""
        from lab5_selection import filter_in_list

        result = filter_in_list(sample_dataframe, 'rock_type', ['Granite', 'Basalt'])

        assert result is not None
        assert len(result) == 4
        assert all(result['rock_type'].isin(['Granite', 'Basalt']))

    def test_filter_multiple_conditions(self, sample_dataframe):
        """Test filtering with multiple conditions."""
        from lab5_selection import filter_multiple_conditions

        result = filter_multiple_conditions(sample_dataframe, 'Granite', 2.0)

        assert result is not None
        assert all(result['rock_type'] == 'Granite')
        assert all(result['grade'] >= 2.0)

    def test_get_unique_values(self, sample_dataframe):
        """Test getting unique values."""
        from lab5_selection import get_unique_values

        result = get_unique_values(sample_dataframe, 'rock_type')

        assert result is not None
        assert len(result) == 3  # Granite, Basalt, Schist

    def test_count_unique_values(self, sample_dataframe):
        """Test counting unique values."""
        from lab5_selection import count_unique_values

        result = count_unique_values(sample_dataframe, 'rock_type')

        assert result is not None
        assert result['Granite'] == 2
        assert result['Basalt'] == 2


# =============================================================================
# Part 3: GroupBy Operations Tests
# =============================================================================

class TestGroupByOperations:
    """Tests for lab5_groupby.py"""

    def test_group_and_mean(self, sample_dataframe):
        """Test groupby mean calculation."""
        from lab5_groupby import group_and_mean

        result = group_and_mean(sample_dataframe, 'rock_type', 'grade')

        assert result is not None
        assert isinstance(result, pd.Series)
        assert 'Granite' in result.index
        # Granite grades: 2.5, 3.2 -> mean = 2.85
        assert abs(result['Granite'] - 2.85) < 0.01

    def test_group_and_count(self, sample_dataframe):
        """Test groupby count."""
        from lab5_groupby import group_and_count

        result = group_and_count(sample_dataframe, 'rock_type')

        assert result is not None
        assert result['Granite'] == 2
        assert result['Basalt'] == 2
        assert result['Schist'] == 1

    def test_group_and_aggregate(self, sample_dataframe):
        """Test groupby with different aggregations."""
        from lab5_groupby import group_and_aggregate

        agg_dict = {'grade': 'mean', 'depth': 'max'}
        result = group_and_aggregate(sample_dataframe, 'rock_type', agg_dict)

        assert result is not None
        assert 'grade' in result.columns
        assert 'depth' in result.columns

    def test_group_multiple_aggregations(self, sample_dataframe):
        """Test multiple aggregations on same column."""
        from lab5_groupby import group_multiple_aggregations

        result = group_multiple_aggregations(sample_dataframe, 'rock_type', 'grade')

        assert result is not None
        assert 'mean' in result.columns
        assert 'std' in result.columns
        assert 'min' in result.columns
        assert 'max' in result.columns

    def test_top_n_per_group(self, larger_dataframe):
        """Test getting top N records per group."""
        from lab5_groupby import top_n_per_group

        result = top_n_per_group(larger_dataframe, 'rock_type', 'grade', n=2)

        assert result is not None
        # Each rock type should have at most 2 records
        for rock_type in result['rock_type'].unique():
            count = len(result[result['rock_type'] == rock_type])
            assert count <= 2

    def test_group_and_transform(self, sample_dataframe):
        """Test groupby transform."""
        from lab5_groupby import group_and_transform

        result = group_and_transform(sample_dataframe, 'rock_type', 'grade')

        assert result is not None
        assert len(result) == len(sample_dataframe)
        # First and third rows are Granite, should have same transform value
        assert result.iloc[0] == result.iloc[2]


# =============================================================================
# Part 4: Descriptive Statistics Tests
# =============================================================================

class TestDescriptiveStatistics:
    """Tests for lab5_statistics.py"""

    def test_get_summary_statistics(self, sample_dataframe):
        """Test describe() summary statistics."""
        from lab5_statistics import get_summary_statistics

        result = get_summary_statistics(sample_dataframe)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'mean' in result.index
        assert 'std' in result.index

    def test_calculate_column_mean(self, sample_dataframe):
        """Test column mean calculation."""
        from lab5_statistics import calculate_column_mean

        result = calculate_column_mean(sample_dataframe, 'grade')

        assert result is not None
        # Mean of [2.5, 1.8, 3.2, 0.9, 2.1] = 2.1
        assert abs(result - 2.1) < 0.01

    def test_calculate_column_std(self, sample_dataframe):
        """Test column standard deviation."""
        from lab5_statistics import calculate_column_std

        result = calculate_column_std(sample_dataframe, 'grade')

        assert result is not None
        assert result > 0

    def test_calculate_min_max(self, sample_dataframe):
        """Test min/max calculation."""
        from lab5_statistics import calculate_min_max

        result = calculate_min_max(sample_dataframe, 'grade')

        assert result is not None
        assert 'min' in result
        assert 'max' in result
        assert result['min'] == 0.9
        assert result['max'] == 3.2

    def test_calculate_percentiles(self, larger_dataframe):
        """Test percentile calculation."""
        from lab5_statistics import calculate_percentiles

        result = calculate_percentiles(larger_dataframe, 'grade', [25, 50, 75])

        assert result is not None
        assert 25 in result
        assert 50 in result
        assert 75 in result
        assert result[25] <= result[50] <= result[75]

    def test_calculate_correlations(self, numeric_only_dataframe):
        """Test correlation matrix calculation."""
        from lab5_statistics import calculate_correlations

        result = calculate_correlations(numeric_only_dataframe)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Diagonal should be 1.0 (self-correlation)
        assert abs(result.loc['x', 'x'] - 1.0) < 0.01
        # x and y should be highly correlated
        assert result.loc['x', 'y'] > 0.8

    def test_find_outliers_iqr(self, dataframe_with_outliers):
        """Test IQR outlier detection."""
        from lab5_statistics import find_outliers_iqr

        result = find_outliers_iqr(dataframe_with_outliers, 'grade')

        assert result is not None
        assert 50.0 in result['grade'].values  # The outlier value

    def test_count_missing_by_column(self, dataframe_with_nulls):
        """Test missing value counting."""
        from lab5_statistics import count_missing_by_column

        result = count_missing_by_column(dataframe_with_nulls)

        assert result is not None
        assert result['grade'] == 2


# =============================================================================
# Part 5: Visualization Tests
# =============================================================================

class TestVisualization:
    """Tests for lab5_visualization.py"""

    def test_create_histogram(self, sample_dataframe):
        """Test histogram creation."""
        from lab5_visualization import create_histogram
        import matplotlib.pyplot as plt

        fig = create_histogram(sample_dataframe, 'grade', bins=10)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_create_scatter_plot(self, sample_dataframe):
        """Test scatter plot creation."""
        from lab5_visualization import create_scatter_plot
        import matplotlib.pyplot as plt

        fig = create_scatter_plot(sample_dataframe, 'depth', 'grade')

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_create_scatter_plot_with_color(self, sample_dataframe):
        """Test scatter plot with color coding."""
        from lab5_visualization import create_scatter_plot
        import matplotlib.pyplot as plt

        fig = create_scatter_plot(
            sample_dataframe, 'depth', 'grade',
            color_column='rock_type'
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_create_box_plot(self, sample_dataframe):
        """Test box plot creation."""
        from lab5_visualization import create_box_plot
        import matplotlib.pyplot as plt

        fig = create_box_plot(sample_dataframe, 'grade', 'rock_type')

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_save_figure(self, sample_dataframe, output_dir):
        """Test saving figure to file."""
        from lab5_visualization import create_histogram, save_figure
        import matplotlib.pyplot as plt

        fig = create_histogram(sample_dataframe, 'grade')
        if fig is None:
            pytest.skip("create_histogram not implemented")

        output_path = output_dir / "test_figure.png"
        save_figure(fig, str(output_path))

        assert output_path.exists(), "Figure should be saved to file"

    def test_create_bar_chart(self, sample_dataframe):
        """Test bar chart creation."""
        from lab5_visualization import create_bar_chart
        import matplotlib.pyplot as plt

        fig = create_bar_chart(sample_dataframe, 'rock_type', 'grade')

        assert fig is not None
        assert isinstance(fig, plt.Figure)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_load_filter_aggregate(self, drilling_data_path):
        """Test complete workflow: load, filter, aggregate."""
        from lab5_dataframe_basics import read_drilling_data
        from lab5_selection import filter_greater_than
        from lab5_groupby import group_and_mean

        if not drilling_data_path.exists():
            pytest.skip("Drilling data file not found")

        # Load data
        df = read_drilling_data(str(drilling_data_path))
        assert df is not None

        # Filter
        filtered = filter_greater_than(df, 'grade', 2.0)
        assert filtered is not None
        assert len(filtered) < len(df)

        # Aggregate
        means = group_and_mean(filtered, 'rock_type', 'grade')
        assert means is not None

    def test_statistics_and_visualization(self, larger_dataframe):
        """Test combining statistics with visualization."""
        from lab5_statistics import find_outliers_iqr
        from lab5_visualization import create_histogram
        import matplotlib.pyplot as plt

        # Find outliers
        outliers = find_outliers_iqr(larger_dataframe, 'grade')

        # Create visualization
        fig = create_histogram(larger_dataframe, 'grade', title='Grade Distribution')

        # Both should work
        if outliers is not None and fig is not None:
            assert isinstance(outliers, pd.DataFrame)
            assert isinstance(fig, plt.Figure)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
