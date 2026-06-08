# dakitlab

<p align="center">
  <b>Professional Tables, Statistical Summaries, Data Validation, and Data Health Reports for Python Dataframes.</b>
</p>

<p align="center">
  Built for Data Analysts, Data Scientists, Researchers, Educators, and Students.
</p>

---

## Overview

`dakitlab` is a Python package designed to reduce repetitive notebook code and help users quickly explore, validate, and present data.

The current release focuses on a powerful `Table` class that provides:

* Professional Plotly-powered tables
* Interactive dataframe viewing
* Statistical summary reports
* Data integrity validation
* Data health assessment reports
* Support for multiple dataframe libraries

Whether you are working in:

* Google Colab
* Jupyter Notebook
* JupyterLab
* Kaggle Notebooks
* VS Code Notebooks

`dakitlab` helps you spend less time writing boilerplate code and more time understanding your data.

---

# Installation

```bash
pip install dakitlab
```

---

# Import

```python
from dakitlab import Table
```

---

# Supported Dataframe Libraries

The `Table` class automatically accepts:

| Library | Supported |
| ------- | --------- |
| Pandas  | ✅         |
| Polars  | ✅         |

Internally, data is converted when necessary so users can work with their preferred dataframe library.

---

# Example Dataset

Examples throughout this README use an environmental monitoring dataset containing:

* Latitude
* Longitude
* PM10
* PM2.5
* Carbon Monoxide
* Nitrogen Dioxide
* Ozone
* Dust
* UV Index
* European AQI
* Hazardous Event

```python
import pandas as pd

df = pd.read_csv("environmental_data.csv")
```

---

# Quick Start

```python
from dakitlab import Table

table = Table(
    df,
    title="Environmental Monitoring Data"
)

table.show()
```

---

# Creating a Table

## Basic Table

```python
table = Table(df)
```

## Table With Title

```python
table = Table(
    df,
    title="Environmental Monitoring Data"
)
```

## Table With Custom Headers

```python
table = Table(
    df,
    header_names=[
        "Latitude",
        "Longitude",
        "PM10",
        "PM2.5",
        "CO",
        "NO₂",
        "Ozone",
        "Dust",
        "UV",
        "AQI",
        "Hazard"
    ]
)
```

---

# Display Methods

## show()

Display the dataframe as a professional Plotly table.

```python
table.show()
```

Custom caption:

```python
table.show(
    caption="Air Quality Monitoring Results"
)
```

---

## display()

Full display control.

```python
table.display(
    filename="environmental_table",
    max_rows=500,
    show_index=False
)
```

---

## interactive()

Displays the dataframe using an interactive notebook table.

```python
table.interactive()
```

Specify rows per page:

```python
table.interactive(
    rows_per_page=50
)
```

---

# Layout Customization

## set_layout()

Control title alignment, dimensions, margins, and column widths.

```python
table.set_layout(
    title="Environmental Monitoring Data",
    title_align="center",
    width=1200,
    height=700
)
```

Advanced example:

```python
table.set_layout(
    width=1400,
    height=800,
    header_height=50,
    cell_height=35,
    margin={
        "l":20,
        "r":20,
        "t":80,
        "b":20
    },
    column_widths=[
        150,150,120,120,120,
        120,120,120,100,100,120
    ]
)
```

---

# Header Styling

## set_header_style()

```python
table.set_header_style(
    fillcolor="#1f2937",
    textcolor="white",
    align="center",
    fontsize=14,
    bold=True
)
```

Supported fonts:

* Arial
* Calibri
* Helvetica
* Times New Roman
* Courier New
* Verdana

---

# Cell Styling

## set_cell_style()

```python
table.set_cell_style(
    fillcolor=["#ffffff", "#f9fafb"],
    textcolor="#111827",
    align="left",
    fontsize=12
)
```

Alternating row colors:

```python
table.set_cell_style(
    fillcolor=[
        "#ffffff",
        "#f3f4f6"
    ]
)
```

---

# Global Styling

## set_global_style()

```python
table.set_global_style(
    paper_bgcolor="#f3f4f6"
)
```

---

# Statistical Summary Reports

The `stats()` method generates descriptive statistics and EDA summaries.

---

## Basic Statistics

```python
table.stats()
```

Returns:

* Count
* Missing values
* Missing %
* Unique values
* Mean
* Standard deviation
* Min
* Max
* Range
* Coefficient of variation

---

## Full Statistics

```python
table.stats(mode="full")
```

Adds:

* Median
* Variance
* Quartiles
* IQR
* Outlier counts
* Outlier percentages
* Skewness
* Distribution shape
* Status indicators

---

## Select Specific Columns By Index

```python
table.stats(
    columns=[2,3,4]
)
```

---

## Select Specific Columns By Name

```python
table.stats(
    columns=[
        "PM10_ug_m3",
        "PM2_5_ug_m3",
        "European_AQI"
    ]
)
```

---

## Full Statistics For Selected Columns

```python
table.stats(
    columns=[
        "PM10_ug_m3",
        "PM2_5_ug_m3"
    ],
    mode="full"
)
```

---

## Round Output

```python
table.stats(
    round_digits=2
)
```

---

# Data Integrity Validation

The `integrity()` method validates data using user-defined rules.

---

## Basic Integrity Check

```python
table.integrity()
```

Uses built-in checks only.

---

## Validate Selected Columns

Using column indexes:

```python
table.integrity(
    columns=[0,1,2]
)
```

Using column names:

```python
table.integrity(
    columns=[
        "Latitude",
        "Longitude"
    ]
)
```

---

# Creating Rules

Rules are defined as a dictionary.

Example:

```python
rules = {

    "Latitude": {
        "required": True,
        "dtype": "numeric",
        "min": -90,
        "max": 90
    },

    "Longitude": {
        "required": True,
        "dtype": "numeric",
        "min": -180,
        "max": 180
    }

}
```

Run:

```python
table.integrity(
    rules=rules
)
```

---

# Supported Rules

## Required Values

```python
{
    "required": True
}
```

---

## Unique Values

```python
{
    "unique": True
}
```

---

## Data Type Validation

```python
{
    "dtype": "numeric"
}
```

Supported:

```python
numeric
text
boolean
date
datetime
```

---

## Numeric Range Validation

```python
{
    "min": 0,
    "max": 100
}
```

---

## Positive Values

```python
{
    "positive": True
}
```

---

## Non-Negative Values

```python
{
    "non_negative": True
}
```

---

## Allowed Values

```python
{
    "allowed": [0,1]
}
```

---

## Minimum Length

```python
{
    "min_length": 3
}
```

---

## Maximum Length

```python
{
    "max_length": 50
}
```

---

## Alphabetic Only

```python
{
    "isalpha": True
}
```

Example:

```text
Andrew
Alice
Bob
```

---

## Numeric Only

```python
{
    "isnumeric": True
}
```

Example:

```text
12345
67890
```

---

## Alphanumeric Only

```python
{
    "isalnum": True
}
```

Example:

```text
ABC123
Student01
```

---

## Regular Expressions

Email validation:

```python
{
    "regex": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
}
```

Phone validation:

```python
{
    "regex": r"^[0-9\-\+\(\) ]+$"
}
```

---

## Allowed Characters

```python
{
    "allowed_chars": "A-Za-z0-9_"
}
```

Allows:

```text
abc
ABC
123
student_01
```

---

# Complete Integrity Example

```python
rules = {

    "Latitude": {
        "required": True,
        "dtype": "numeric",
        "min": -90,
        "max": 90
    },

    "Longitude": {
        "required": True,
        "dtype": "numeric",
        "min": -180,
        "max": 180
    },

    "Hazardous_Event": {
        "allowed": [0,1]
    },

    "Station_Name": {
        "required": True,
        "min_length": 3,
        "max_length": 50
    }

}

table.integrity(
    rules=rules
)
```

---

# Data Health Reports

The `data_health()` method provides a high-level overview of dataset quality.

```python
table.data_health()
```

---

## What Is Included?

* Dataset health score
* Missing value analysis
* Missing rows report
* Duplicate row detection
* Rows requiring attention
* Severity classification

---

## Limit Problem Rows Displayed

```python
table.data_health(
    max_problem_rows=25
)
```

---

## Hide Problem Rows

```python
table.data_health(
    show_problem_rows=False
)
```

---

# Complete Workflow Example

```python
from dakitlab import Table

table = Table(
    df,
    title="Environmental Monitoring Data"
)

table.show()

table.stats(
    columns=[
        "PM10_ug_m3",
        "PM2_5_ug_m3",
        "European_AQI"
    ],
    mode="full"
)

rules = {
    "Latitude": {
        "required": True,
        "min": -90,
        "max": 90
    },

    "Longitude": {
        "required": True,
        "min": -180,
        "max": 180
    }
}

table.integrity(rules=rules)

table.data_health()
```

---

# Current Public Methods

| Method             | Description                 |
| ------------------ | --------------------------- |
| Table()            | Create a table object       |
| show()             | Display styled table        |
| display()          | Advanced table display      |
| interactive()      | Interactive dataframe view  |
| set_layout()       | Layout customization        |
| set_header_style() | Header customization        |
| set_cell_style()   | Cell customization          |
| set_global_style() | Global styling              |
| stats()            | Statistical summary reports |
| integrity()        | Rule-based validation       |
| data_health()      | Dataset health assessment   |

---

# Roadmap

Planned future classes:

* Summary
* CompareFrames
* Cleaner
* SchemaValidator
* QuickPlot
* CorrelationMap
* DistributionGrid
* Report
* Snapshot
* Profiler

---

# License

MIT License

---

# Author

Andrew Benyeogor Osenwe

Built for practical data analysis, exploratory data analysis, and notebook productivity.
