# Lab 5: Data Analysis with pandas

**Course:** GGY3601 - Introduction to Programming for Geologists
**Weight:** 20%
**Estimated Time:** 3-4 hours

## Overview

This lab introduces you to pandas, Python's powerful data analysis library. You will learn to create and manipulate DataFrames, select and filter data, perform groupby operations, generate descriptive statistics, and create basic visualizations. These skills are fundamental for geological data analysis and form the foundation for more advanced data science workflows.

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

### 1. Accept the Assignment

Click the GitHub Classroom assignment link provided by your instructor.

### 2. Clone Your Repository

```bash
git clone <your-repository-url>
cd ggy3601-lab05-pandas-analysis
```

### 3. Install Dependencies

```bash
pip install pandas matplotlib pytest
```

### 4. Run the Tests

```bash
pytest tests/visible/ -v
```

## Lab Structure

### Part 1: DataFrame Basics (20 points)
**File:** `src/lab5_dataframe_basics.py`

Learn to create DataFrames from dictionaries and CSV files:
- Creating DataFrames from dictionaries
- Reading CSV files with `pd.read_csv()`
- Understanding DataFrame structure (shape, columns, dtypes)
- Basic DataFrame inspection methods

### Part 2: Data Selection (20 points)
**File:** `src/lab5_selection.py`

Master data selection and filtering techniques:
- Column selection with bracket notation
- Row selection with `loc` and `iloc`
- Boolean indexing for filtering
- Combining multiple conditions

### Part 3: GroupBy Operations (20 points)
**File:** `src/lab5_groupby.py`

Learn to aggregate data by categories:
- Basic groupby operations
- Multiple aggregation functions
- Named aggregations
- Transformations within groups

### Part 4: Descriptive Statistics (20 points)
**File:** `src/lab5_statistics.py`

Generate statistical summaries of your data:
- Using `describe()` for quick statistics
- Individual statistical methods (mean, std, min, max)
- Correlation analysis
- Handling missing data in statistics

### Part 5: Basic Visualization (20 points)
**File:** `src/lab5_visualization.py`

Create visualizations from DataFrame data:
- Histograms for distribution analysis
- Scatter plots for relationships
- Box plots for comparing groups
- Saving figures to files

## Dataset

You will work with drilling data in `data/drilling_data.csv` containing:
- `sample_id`: Unique identifier for each sample
- `hole_id`: Drillhole identifier
- `rock_type`: Lithological classification
- `grade`: Ore grade (g/t or %)
- `depth`: Sample depth (meters)
- `mass`: Sample mass (kg)
- `volume`: Sample volume (cm3)
- `location`: Site location
- `analyst`: Analyst who processed the sample

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

## Tips for Success

1. **Start with the basics:** Make sure you understand DataFrame structure before moving to complex operations
2. **Use the REPL:** Test your code interactively before implementing functions
3. **Check data types:** Many pandas errors come from unexpected data types
4. **Read error messages:** pandas error messages are usually informative
5. **Explore the data:** Use `head()`, `info()`, and `describe()` frequently

## Academic Integrity

This is an individual assignment. You may discuss concepts with classmates, but all code must be your own. Refer to the course syllabus for the academic integrity policy.

## Support

- **Office Hours:** See course schedule
- **Discussion Forum:** Post questions on the course forum
- **Email:** Contact your instructor for complex issues

---

Good luck with your data analysis!
