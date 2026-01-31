# Lab 5: Data Analysis with pandas

**Course:** GGY3601 - Introduction to Programming for Geologists
**Weight:** 20%
**Estimated Time:** 3-4 hours

## Overview

This lab introduces you to pandas, Python's powerful data analysis library. You will learn to create and manipulate DataFrames, select and filter data, perform groupby operations, generate descriptive statistics, and create basic visualizations.

## Your Assignment Parameters

Your personalized assignment uses the following parameters:

- **Analysis Columns:** {analysis_columns}
- **GroupBy Column:** {groupby_column}
- **Filter Threshold:** {filter_threshold}
- **Number of Records:** {num_records}

These values are unique to your assignment and will be used in your data analysis tasks.

## Learning Outcomes

By completing this lab, you will be able to:

- **LO5.1:** Create and manipulate pandas DataFrames from various data sources
- **LO5.2:** Select and filter data effectively using loc, iloc, and boolean indexing
- **LO5.3:** Perform groupby operations and aggregations on geological datasets
- **LO5.4:** Generate descriptive statistics to summarize data distributions
- **LO5.5:** Create basic visualizations using matplotlib integration with pandas

## Prerequisites

- Completion of Labs 1-4
- Understanding of Python data structures (lists, dictionaries)
- Basic familiarity with CSV file operations
- Python 3.11+ with pandas and matplotlib installed

## Getting Started

### 1. Clone Your Repository

```bash
git clone <your-repository-url>
cd ggy3601-lab05-pandas-analysis
```

### 2. Install Dependencies

```bash
pip install pandas matplotlib pytest
```

### 3. Run the Tests

```bash
pytest tests/visible/ -v
```

## Lab Structure

### Part 1: DataFrame Basics (20 points)
**File:** `src/lab5_dataframe_basics.py`

Complete the following functions:
- `create_sample_dataframe()` - Create a DataFrame from a dictionary
- `read_drilling_data()` - Load CSV data into a DataFrame
- `get_dataframe_info()` - Extract basic DataFrame information
- `display_first_n_rows()` - Return the first n rows

### Part 2: Data Selection (20 points)
**File:** `src/lab5_selection.py`

Using your analysis columns ({analysis_columns}):
- `select_columns()` - Select specific columns
- `select_rows_by_index()` - Select rows using iloc
- `filter_by_threshold()` - Filter data where grade > {filter_threshold}
- `filter_multiple_conditions()` - Apply multiple filter conditions

### Part 3: GroupBy Operations (20 points)
**File:** `src/lab5_groupby.py`

Using your groupby column ({groupby_column}):
- `group_and_mean()` - Calculate mean by group
- `group_and_aggregate()` - Multiple aggregations per group
- `group_and_count()` - Count records per group
- `top_n_per_group()` - Get top N records per group

### Part 4: Descriptive Statistics (20 points)
**File:** `src/lab5_statistics.py`

- `get_summary_statistics()` - Use describe() on numeric columns
- `calculate_correlations()` - Compute correlation matrix
- `find_outliers()` - Identify statistical outliers
- `calculate_percentiles()` - Compute custom percentiles

### Part 5: Basic Visualization (20 points)
**File:** `src/lab5_visualization.py`

- `create_histogram()` - Create a histogram of grade distribution
- `create_scatter_plot()` - Plot grade vs depth relationship
- `create_box_plot()` - Compare grades across rock types
- `save_figure()` - Save plot to file

## Dataset

Your dataset contains {num_records} drilling records in `data/drilling_data.csv`:
- `sample_id`: Unique identifier for each sample
- `hole_id`: Drillhole identifier
- `rock_type`: Lithological classification
- `grade`: Ore grade (g/t or %)
- `depth`: Sample depth (meters)
- `mass`: Sample mass (kg)
- `volume`: Sample volume (cm3)
- `location`: Site location
- `analyst`: Analyst who processed the sample

## Specific Tasks

### Task 1: Data Loading and Exploration
Load the drilling data and verify it contains {num_records} records. Explore the DataFrame structure and data types.

### Task 2: Column Analysis
Focus your analysis on these columns: {analysis_columns}. Calculate statistics for each and identify patterns.

### Task 3: Grouped Analysis
Group the data by {groupby_column} and calculate summary statistics for each group.

### Task 4: Filtering
Filter the dataset to include only samples with grade > {filter_threshold}. Analyze how this affects the data distribution.

### Task 5: Visualization
Create visualizations showing the relationship between your analysis columns and grouped by {groupby_column}.

## Requirements

### Code Quality
- Follow PEP 8 style guidelines
- Include docstrings for all functions
- Use meaningful variable names
- Add comments for complex operations

### Testing
- All visible tests must pass
- Functions should handle edge cases
- Proper error handling for invalid inputs

## Submission

1. Complete all functions in the `src/` files
2. Ensure all visible tests pass locally
3. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Complete Lab 5"
   git push
   ```
4. Check the Actions tab for automated test results

## Grading Rubric

| Component | Points | Description |
|-----------|--------|-------------|
| DataFrame Basics | 20 | Creating and inspecting DataFrames |
| Data Selection | 20 | Filtering and selecting data |
| GroupBy Operations | 20 | Aggregating data by groups |
| Descriptive Statistics | 20 | Statistical analysis |
| Visualization | 20 | Creating basic plots |
| **Total** | **100** | |

## Resources

- [pandas Documentation](https://pandas.pydata.org/docs/)
- [pandas Getting Started Tutorials](https://pandas.pydata.org/docs/getting_started/index.html)
- [Matplotlib Pyplot Tutorial](https://matplotlib.org/stable/tutorials/pyplot.html)
- Course lecture slides on pandas

## Academic Integrity

This is an individual assignment. You may discuss concepts with classmates, but all code must be your own.

---

Good luck with your data analysis!
