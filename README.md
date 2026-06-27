# dakitlab

<p align="center">
  <b>Professional Tables, Statistical Summaries, Data Validation, and Data Health Reports for Python DataFrames.</b>
</p>

<p align="center">
  Built for Data Analysts, Data Scientists, Researchers, Educators, and Students.
</p>

---

## Overview

`dakitlab` is a Python package that eliminates repetitive notebook boilerplate so you can explore, validate, and present data faster.

The current release centers on a powerful `Table` class that provides:

- Professional HTML-rendered tables with sticky headers and horizontal scroll
- Interactive, paginated DataFrame viewing
- Statistical summary reports (fast and full modes)
- Rule-based data integrity validation
- Data health assessment with health scoring
- CSV sampling and download
- Support for six DataFrame libraries out of the box

Works seamlessly in:

- Google Colab
- Jupyter Notebook / JupyterLab
- Kaggle Notebooks
- VS Code Notebooks

---

## Installation

```bash
pip install dakitlab
```

---

## Import

```python
from dakitlab import Table
```

---

## Supported DataFrame Libraries

`Table` automatically detects and converts any of the following — no manual conversion needed.

| Library | Supported |
| ------- | --------- |
| Pandas  | ✅        |
| Polars  | ✅        |
| Dask    | ✅        |
| PySpark | ✅        |
| cuDF    | ✅        |
| Vaex    | ✅        |

---

## Example Dataset

Examples throughout this README use an environmental monitoring dataset with these columns:

`Latitude`, `Longitude`, `PM10`, `PM2.5`, `Carbon Monoxide`, `Nitrogen Dioxide`, `Ozone`, `Dust`, `UV Index`, `European AQI`, `Hazardous Event`

```python
import pandas as pd

df = pd.read_csv("environmental_data.csv")
```

---

## Quick Start

```python
from dakitlab import Table

table = Table(df, title="Environmental Monitoring Data")
table.show()
```

---

## Creating a Table

```python
# Basic
table = Table(df)

# With title
table = Table(df, title="Environmental Monitoring Data")

# With custom column headers
table = Table(
    df,
    header_names=[
        "Latitude", "Longitude", "PM10", "PM2.5",
        "CO", "NO₂", "Ozone", "Dust", "UV", "AQI", "Hazard"
    ]
)
```

---

## Display Methods

### show()

Render the table inline. Accepts an optional one-shot caption that overrides the table title for that render only.

```python
table.show()
table.show(caption="Air Quality Monitoring Results")
```

---

### display()

Full display control. Renders the table with Download CSV and Download HTML buttons.

```python
table.display(
    filename="environmental_table",
    caption="Q4 Results",
    max_rows=500,
    show_index=False
)
```

| Parameter    | Type       | Default        | Description                                          |
| ------------ | ---------- | -------------- | ---------------------------------------------------- |
| `filename`   | `str`      | `"newtable"`   | Base name for downloaded CSV and HTML files          |
| `caption`    | `str/None` | `None`         | One-shot title override for this render only         |
| `max_rows`   | `int/None` | `1000`         | Row limit; `None` renders all rows                   |
| `show_index` | `bool`     | `False`        | Prepend the DataFrame index as a column              |

---

### interactive()

Paginated interactive view. Uses Colab's built-in `DataTable` in Google Colab and falls back to a plain HTML table elsewhere.

```python
table.interactive()
table.interactive(rows_per_page=50)
```

---

### download()

Download the DataFrame (or a sample) directly as a CSV file. Triggers an immediate browser download with no UI.

```python
# Full dataset
table.download()

# First 100 rows
table.download(n_rows=100)

# 100 randomly sampled rows
table.download(n_rows=100, randomize=True)

# Reproducible random sample
table.download(n_rows=100, randomize=True, seed=42)

# Custom filename
table.download(filename="q4_sample", n_rows=500, randomize=True)
```

| Parameter   | Type       | Default      | Description                                          |
| ----------- | ---------- | ------------ | ---------------------------------------------------- |
| `filename`  | `str`      | `"download"` | Base filename (`.csv` appended automatically)        |
| `n_rows`    | `int/None` | `None`       | Rows to include; `None` downloads the full dataset   |
| `randomize` | `bool`     | `False`      | Sample randomly instead of taking the first `n_rows` |
| `seed`      | `int/None` | `None`       | Random seed for reproducible samples                 |

If `n_rows` exceeds the number of rows in the DataFrame, a warning is raised and all rows are downloaded.

---

## Layout Customization

### set_layout()

Control title text and alignment, scroll height, sticky header behaviour, and column widths. Returns `self` for chaining.
Note: column_width also accept dictionaries e.g {2: 100, 6:400} means you want the make column 2 and 6 to be 100pts and 600pts wide respectively

```python
table.set_layout(
    title="Environmental Monitoring Data",
    title_align="center",       # "left" | "center" | "right"
    max_height="400px",         # any CSS height, e.g. "80vh"
    sticky_header=True,
    column_widths=[150, 150, 120, 120, 120, 120, 120, 120, 100, 100, 120]
)
```

Set `max_height="none"` to disable vertical scrolling entirely.

---

## Styling

All style setters return `self` so they can be chained with each other and with display calls.

```python
table.set_header_style(fillcolor="#1a237e").set_cell_style(fontsize=12).show()
```

### set_header_style()

Customise the appearance of the header row.

```python
table.set_header_style(
    fillcolor="#1f2937",
    textcolor="#ffffff",
    align="left",           # "left" | "center" | "right"
    fontsize=13,
    font_family="Arial",
    bold=True,
    italic=False
)
```

### set_cell_style()

Customise the appearance of data cells. Pass a two-element list to `fillcolor` for alternating row colours.

```python
# Single background colour
table.set_cell_style(
    fillcolor="#ffffff",
    textcolor="#111827",
    align="left",
    linecolor="#e5e7eb",
    fontsize=12,
    font_family="Arial",
    bold=False,
    italic=False
)

# Alternating row colours
table.set_cell_style(fillcolor=["#ffffff", "#f3f4f6"])
```

### set_global_style()

Set the canvas background colour.

```python
table.set_global_style(paper_bgcolor="#f3f4f6")
```

---

## Statistical Summary Reports

### stats()

Generates descriptive statistics for all numeric columns (or a subset). Returns the summary as a DataFrame and renders an HTML report inline.

#### Fast mode (default)

Returns: count, missing, missing %, unique, mean, std, min, max, range, CV, zero-variance flag, status.

```python
table.stats()
```

#### Full mode

Adds: median, variance, Q1, Q3, IQR, lower/upper fence bounds, outlier count, outlier %, skewness, distribution shape.

```python
table.stats(mode="full")
```

#### Select columns

```python
# By index
table.stats(columns=[2, 3, 4])

# By name
table.stats(columns=["PM10_ug_m3", "PM2_5_ug_m3", "European_AQI"])

# Full mode on a subset
table.stats(columns=["PM10_ug_m3", "PM2_5_ug_m3"], mode="full")
```

#### Round output

```python
table.stats(round_digits=2)
```

#### Parameters

| Parameter     | Type           | Default               | Description                        |
| ------------- | -------------- | --------------------- | ---------------------------------- |
| `columns`     | `list/None`    | `None`                | Column names or indexes to analyse |
| `file_name`   | `str`          | `"summary_stats.html"`| HTML report download filename      |
| `mode`        | `str`          | `"fast"`              | `"fast"` or `"full"`               |
| `round_digits`| `int`          | `3`                   | Decimal places for numeric output  |
| `return_data` | `bool`         | `True`                | Return the summary DataFrame       |

---

## Data Integrity Validation

### integrity()

Validates each column against optional user-defined rules and renders a per-column HTML report. Returns the report as a DataFrame.

#### Basic check (built-in checks only)

```python
table.integrity()
```

#### Select columns

```python
table.integrity(columns=[0, 1, 2])
table.integrity(columns=["Latitude", "Longitude"])
```

#### With rules

Rules are a dict mapping column names to rule sets.

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

table.integrity(rules=rules)
```

---

### Supported Rules

#### Shared (numeric and text)

| Rule       | Type       | Description                              |
| ---------- | ---------- | ---------------------------------------- |
| `required` | `bool`     | Flag rows where this column is null      |
| `unique`   | `bool`     | Flag if any duplicate values exist       |
| `dtype`    | `str`      | Expected type: `numeric`, `text`, `bool`, `date`, `datetime` |
| `allowed`  | `list`     | Whitelist of valid values                |
| `regex`    | `str`      | Values must match this regex pattern     |
| `custom`   | `callable` | `fn(value) → bool` — `True` means valid  |

#### Numeric only

| Rule           | Type    | Description                         |
| -------------- | ------- | ----------------------------------- |
| `positive`     | `bool`  | All values must be > 0              |
| `non_negative` | `bool`  | All values must be ≥ 0              |
| `min`          | `float` | Lower bound (inclusive)             |
| `max`          | `float` | Upper bound (inclusive)             |

#### Text only

| Rule            | Type  | Description                                        |
| --------------- | ----- | -------------------------------------------------- |
| `not_empty`     | `bool`| No blank or whitespace-only strings                |
| `isalpha`       | `bool`| Only alphabetic characters                         |
| `isnumeric`     | `bool`| Only digit characters                              |
| `isalnum`       | `bool`| Only alphanumeric characters                       |
| `allowed_chars` | `str` | Regex character class, e.g. `"A-Za-z0-9_"`        |
| `min_length`    | `int` | Minimum string length                              |
| `max_length`    | `int` | Maximum string length                              |

#### Custom rule example

```python
def is_valid_score(value):
    return 0 <= value <= 100

rules = {"Score": {"custom": is_valid_score}}
table.integrity(rules=rules)
```

#### Complete example

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
        "allowed": [0, 1]
    },
    "Station_Name": {
        "required": True,
        "not_empty": True,
        "min_length": 3,
        "max_length": 50
    },
    "Email": {
        "regex": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    }
}

table.integrity(rules=rules)
```

#### Parameters

| Parameter      | Type        | Default                   | Description                              |
| -------------- | ----------- | ------------------------- | ---------------------------------------- |
| `columns`      | `list/None` | `None`                    | Column names or indexes to validate      |
| `rules`        | `dict/None` | `None`                    | Validation rules per column              |
| `file_name`    | `str`       | `"integrity_report.html"` | HTML report download filename            |
| `max_examples` | `int`       | `5`                       | Sample values shown per column           |
| `return_data`  | `bool`      | `True`                    | Return the integrity DataFrame           |

---

## Data Health Reports

### data_health()

Provides a high-level dataset quality overview including a health score (0–100), missing value analysis, duplicate row detection, and a prioritised attention table with severity ratings.

```python
table.data_health()
```

After calling `data_health()`, flagged rows are available as a DataFrame:

```python
table.problem_rows
```

#### Options

```python
# Custom report title
table.data_health(title="Q4 Dataset Quality Review")

# Limit problem rows shown
table.data_health(max_problem_rows=25)

# Hide the problem rows table
table.data_health(show_problem_rows=False)
```

#### Parameters

| Parameter          | Type       | Default                  | Description                                      |
| ------------------ | ---------- | ------------------------ | ------------------------------------------------ |
| `title`            | `str`      | `"Data Health Overview"` | Report heading                                   |
| `file_name`        | `str`      | `"health_report.html"`   | HTML report download filename                    |
| `show_problem_rows`| `bool`     | `True`                   | Render the problem rows table inline             |
| `max_problem_rows` | `int/None` | `None`                   | Cap on problem rows shown; `None` shows all      |

---

## Converting to Pandas

`Table.to_pandas()` is a static utility that converts any supported DataFrame to pandas.

```python
pdf = Table.to_pandas(polars_df)
pdf = Table.to_pandas(dask_df)
pdf = Table.to_pandas(spark_df)
```

---

## Complete Workflow Example

```python
from dakitlab import Table

table = Table(df, title="Environmental Monitoring Data")

# Display the table
table.show()

# Download a reproducible 500-row sample
table.download(filename="env_sample", n_rows=500, randomize=True, seed=0)

# Statistical summary for selected columns
table.stats(
    columns=["PM10_ug_m3", "PM2_5_ug_m3", "European_AQI"],
    mode="full"
)

# Integrity validation
rules = {
    "Latitude":  {"required": True, "min": -90,  "max": 90},
    "Longitude": {"required": True, "min": -180, "max": 180}
}
table.integrity(rules=rules)

# Health report
table.data_health()

# Inspect flagged rows
table.problem_rows
```

---

## Public API Reference

| Method               | Description                                              |
| -------------------- | -------------------------------------------------------- |
| `Table()`            | Create a Table from any supported DataFrame              |
| `show()`             | Render the table inline                                  |
| `display()`          | Render with full control and download buttons            |
| `interactive()`      | Paginated interactive DataFrame view                     |
| `download()`         | Download the full dataset or a CSV sample                |
| `set_layout()`       | Configure title, scroll height, sticky header, widths    |
| `set_header_style()` | Customise header appearance                              |
| `set_cell_style()`   | Customise cell appearance                                |
| `set_global_style()` | Set canvas background colour                             |
| `stats()`            | Quantitative summary report                              |
| `integrity()`        | Rule-based column validation report                      |
| `data_health()`      | Dataset health assessment report                         |
| `Table.to_pandas()`  | Convert any supported DataFrame to pandas                |
| `Table.help()`       | Print a concise method reference in the console          |

---

## Roadmap

Planned future classes:

- `Summary`
- `CompareFrames`
- `Cleaner`
- `SchemaValidator`
- `QuickPlot`
- `CorrelationMap`
- `DistributionGrid`
- `Report`
- `Snapshot`
- `Profiler`

---

## License

MIT License

---

## Author

Andrew Benyeogor Osenwe

Built for practical data analysis, exploratory data analysis, and notebook productivity.