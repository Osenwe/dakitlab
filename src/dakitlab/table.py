from __future__ import annotations

from typing import Literal, Any
import base64
import html
import warnings

import pandas as pd
import polars as pl
import plotly.graph_objects as go
from IPython.display import display, HTML


AlignValue = Literal["left", "center", "right"]

FontFamily = Literal[
    "Arial",
    "Calibri",
    "Helvetica",
    "Times New Roman",
    "Courier New",
    "Verdana",
]


class Table:
    def __init__(
        self,
        dataframe: Any,
        header_names: list[str] | None = None,
        title: str | None = None,
    ) -> None:
        self.original_dataframe = dataframe
        self.dataframe = self.to_pandas(dataframe).copy()

        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Nothing to display.")

        if header_names is not None:
            self._validate_header_names(header_names, self.dataframe)

        self.header_names = header_names
        self.title = title
        self.problem_rows: pd.DataFrame | None = None

        self.margin = dict(l=20, r=20, t=60, b=20)
        self.title_align = 0.5

        self.width = 1200
        self.height = 700

        self.header_fillcolor = "#263238"
        self.header_textcolor = "#ffffff"
        self.header_align: AlignValue = "left"

        self.cell_fillcolor = ["#ffffff", "#fafafa"]
        self.cell_textcolor = "#111827"
        self.cell_align: AlignValue = "left"

        self.paper_bgcolor = "#f3f4f6"

        self.header_fontsize = 13
        self.cell_fontsize = 11

        self.header_height = 38
        self.cell_height = 32

        self.header_font_family: FontFamily = "Arial"
        self.cell_font_family: FontFamily = "Arial"

        self.header_bold = True
        self.cell_bold = False

        self.header_italic = False
        self.cell_italic = False

        self.header_linecolor = "#263238"
        self.cell_linecolor = "#e5e7eb"

        self.column_widths: list[int] | None = None

    # ---------------------------------------------------------
    # Layout and style methods
    # ---------------------------------------------------------

    def set_layout(
        self,
        title: str | None = None,
        title_align: AlignValue | None = None,
        width: int | None = None,
        height: int | None = None,
        header_height: int | None = None,
        cell_height: int | None = None,
        margin: dict[str, int] | None = None,
        column_widths: list[int] | None = None,
    ) -> None:
        if title is not None:
            if not isinstance(title, str):
                raise TypeError("title must be a string.")
            self.title = title

        if title_align is not None:
            self._validate_align(title_align, "title_align")
            self.title_align = {
                "left": 0.05,
                "center": 0.5,
                "right": 0.95,
            }[title_align]

        if width is not None:
            self._validate_positive_int(width, "width")
            self.width = width

        if height is not None:
            self._validate_positive_int(height, "height")
            self.height = height

        if header_height is not None:
            self._validate_positive_int(header_height, "header_height")
            self.header_height = header_height

        if cell_height is not None:
            self._validate_positive_int(cell_height, "cell_height")
            self.cell_height = cell_height

        if margin is not None:
            self._validate_margin(margin)
            self.margin = margin

        if column_widths is not None:
            self._validate_column_widths(column_widths)
            self.column_widths = column_widths

    def set_global_style(self, paper_bgcolor: str | None = None) -> None:
        if paper_bgcolor is not None:
            self._validate_color(paper_bgcolor, "paper_bgcolor")
            self.paper_bgcolor = paper_bgcolor

    def set_header_style(
        self,
        fillcolor: str | list[str] | None = None,
        textcolor: str | None = None,
        align: AlignValue | None = None,
        linecolor: str | None = None,
        fontsize: int | None = None,
        font_family: FontFamily | None = None,
        bold: bool | None = None,
        italic: bool | None = None,
    ) -> None:
        if fillcolor is not None:
            self._validate_color_or_list(fillcolor, "header_fillcolor")
            self.header_fillcolor = fillcolor

        if textcolor is not None:
            self._validate_color(textcolor, "header_textcolor")
            self.header_textcolor = textcolor

        if align is not None:
            self._validate_align(align, "header_align")
            self.header_align = align

        if linecolor is not None:
            self._validate_color(linecolor, "header_linecolor")
            self.header_linecolor = linecolor

        if fontsize is not None:
            self._validate_positive_int(fontsize, "header_fontsize")
            self.header_fontsize = fontsize

        if font_family is not None:
            self._validate_font_family(font_family, "header_font_family")
            self.header_font_family = font_family

        if bold is not None:
            self._validate_bool(bold, "header_bold")
            self.header_bold = bold

        if italic is not None:
            self._validate_bool(italic, "header_italic")
            self.header_italic = italic

    def set_cell_style(
        self,
        fillcolor: str | list[str] | None = None,
        textcolor: str | None = None,
        align: AlignValue | None = None,
        linecolor: str | None = None,
        fontsize: int | None = None,
        font_family: FontFamily | None = None,
        bold: bool | None = None,
        italic: bool | None = None,
    ) -> None:
        if fillcolor is not None:
            self._validate_color_or_list(fillcolor, "cell_fillcolor")
            self.cell_fillcolor = fillcolor

        if textcolor is not None:
            self._validate_color(textcolor, "cell_textcolor")
            self.cell_textcolor = textcolor

        if align is not None:
            self._validate_align(align, "cell_align")
            self.cell_align = align

        if linecolor is not None:
            self._validate_color(linecolor, "cell_linecolor")
            self.cell_linecolor = linecolor

        if fontsize is not None:
            self._validate_positive_int(fontsize, "cell_fontsize")
            self.cell_fontsize = fontsize

        if font_family is not None:
            self._validate_font_family(font_family, "cell_font_family")
            self.cell_font_family = font_family

        if bold is not None:
            self._validate_bool(bold, "cell_bold")
            self.cell_bold = bold

        if italic is not None:
            self._validate_bool(italic, "cell_italic")
            self.cell_italic = italic

    # ---------------------------------------------------------
    # Display methods
    # ---------------------------------------------------------

    def interactive(self, rows_per_page: int = 20):
        self._validate_positive_int(rows_per_page, "rows_per_page")

        try:
            from google.colab import data_table

            return data_table.DataTable(
                self.dataframe,
                include_index=False,
                num_rows_per_page=rows_per_page,
            )

        except Exception:
            display(HTML(self.dataframe.to_html(index=False)))

    def show(self, caption: str | None = None) -> None:
        self.display(caption=caption)

    def display(
        self,
        dataframe: pd.DataFrame | None = None,
        filename: str = "newtable",
        caption: str | None = None,
        max_rows: int | None = 1000,
        show_index: bool = False,
    ) -> None:
        self._validate_filename(filename)

        df = self.dataframe.copy() if dataframe is None else self.to_pandas(dataframe).copy()

        if df.empty:
            raise ValueError("Dataframe is empty. Nothing to display.")

        if max_rows is not None:
            self._validate_positive_int(max_rows, "max_rows")

        if not isinstance(show_index, bool):
            raise TypeError("show_index must be True or False.")

        if show_index:
            df = df.reset_index()

        if max_rows is not None and len(df) > max_rows:
            warnings.warn(
                f"Only showing the first {max_rows} rows. "
                "Use interactive() for larger datasets."
            )
            df = df.head(max_rows)

        headers = self.header_names or list(df.columns)

        if self.header_names is not None:
            self._validate_header_names(self.header_names, df)

        headers = self._format_values(
            headers,
            bold=self.header_bold,
            italic=self.cell_italic,
        )

        values = [df[col].tolist() for col in df.columns]

        if self.cell_bold or self.cell_italic:
            values = [
                self._format_values(
                    column,
                    bold=self.cell_bold,
                    italic=self.cell_italic,
                )
                for column in values
            ]

        row_colors = self._make_row_colors(len(df))

        fig = go.Figure(
            data=[
                go.Table(
                    columnwidth=self.column_widths,
                    header=dict(
                        values=headers,
                        fill_color=self.header_fillcolor,
                        font=dict(
                            color=self.header_textcolor,
                            size=self.header_fontsize,
                            family=self.header_font_family,
                        ),
                        align=self.header_align,
                        height=self.header_height,
                        line_color=self.header_linecolor,
                    ),
                    cells=dict(
                        values=values,
                        fill_color=[row_colors],
                        font=dict(
                            color=self.cell_textcolor,
                            size=self.cell_fontsize,
                            family=self.cell_font_family,
                        ),
                        align=self.cell_align,
                        height=self.cell_height,
                        line_color=self.cell_linecolor,
                    ),
                )
            ]
        )

        title_text = caption or self.title

        layout_kwargs = dict(
            width=self.width,
            height=self.height,
            margin=self.margin,
            paper_bgcolor=self.paper_bgcolor,
        )

        if title_text:
            layout_kwargs["title"] = dict(
                text=title_text,
                x=self.title_align,
                xanchor=self._get_title_xanchor(),
            )

        fig.update_layout(**layout_kwargs)

        fig.show(
            config={
                "toImageButtonOptions": {
                    "filename": filename,
                }
            }
        )

    # ---------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------

    def stats(
      self,
      columns: list[int | str] | None = None,
      file_name: str = "summary_stats.html",
      mode: Literal["fast", "full"] = "fast",
      round_digits: int = 3,
      return_data: bool = True,) -> pd.DataFrame | None:

        self._validate_filename(file_name)

        if mode not in ["fast", "full"]:
            raise ValueError("mode must be either 'fast' or 'full'.")

        selected_cols = self._resolve_columns(columns)

        df_pl = self._to_polars(self.dataframe)

        df_pl = df_pl.select(selected_cols)

        numeric_cols = [
            col for col, dtype in df_pl.schema.items()
            if dtype.is_numeric()
        ]

        if not numeric_cols:
            display(HTML("<h3>No quantitative columns found.</h3>"))
            return pd.DataFrame() if return_data else None

        num_df = df_pl.select(numeric_cols)
        total_rows = num_df.height

        exprs = []

        for col in numeric_cols:
            exprs.extend([
                pl.col(col).count().alias(f"{col}__Count"),
                pl.col(col).null_count().alias(f"{col}__Missing"),
                pl.col(col).mean().alias(f"{col}__Mean"),
                pl.col(col).std().alias(f"{col}__Std"),
                pl.col(col).min().alias(f"{col}__Min"),
                pl.col(col).max().alias(f"{col}__Max"),
                pl.col(col).n_unique().alias(f"{col}__Unique"),
            ])

            if mode == "full":
                exprs.extend([
                    pl.col(col).median().alias(f"{col}__Median"),
                    pl.col(col).var().alias(f"{col}__Variance"),
                    pl.col(col).quantile(0.25).alias(f"{col}__Q1"),
                    pl.col(col).quantile(0.75).alias(f"{col}__Q3"),
                    pl.col(col).skew().alias(f"{col}__Skewness"),
                ])

        stats = num_df.select(exprs).to_dicts()[0]

        rows = []

        for col in numeric_cols:
            count = stats.get(f"{col}__Count")
            missing = stats.get(f"{col}__Missing")
            mean = stats.get(f"{col}__Mean")
            std = stats.get(f"{col}__Std")
            min_val = stats.get(f"{col}__Min")
            max_val = stats.get(f"{col}__Max")
            unique = stats.get(f"{col}__Unique")

            row = {
                "Column": col,
                "Count": count,
                "Missing": missing,
                "Missing %": (missing / total_rows * 100) if total_rows else 0,
                "Unique": unique,
                "Mean": mean,
                "Std": std,
                "Min": min_val,
                "Max": max_val,
                "Range": max_val - min_val if min_val is not None and max_val is not None else None,
                "CV": std / mean if mean not in [None, 0] and std is not None else None,
                "Zero Variance": std == 0 if std is not None else False,
            }

            if mode == "full":
                q1 = stats.get(f"{col}__Q1")
                q3 = stats.get(f"{col}__Q3")
                iqr = q3 - q1 if q1 is not None and q3 is not None else None

                row.update({
                    "Median": stats.get(f"{col}__Median"),
                    "Variance": stats.get(f"{col}__Variance"),
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": iqr,
                    "Lower Bound": q1 - 1.5 * iqr if iqr is not None else None,
                    "Upper Bound": q3 + 1.5 * iqr if iqr is not None else None,
                    "Skewness": stats.get(f"{col}__Skewness"),
                })

            rows.append(row)

        summary_df = pd.DataFrame(rows)

        if mode == "full":
            outlier_exprs = []

            for _, row in summary_df.iterrows():
                col = row["Column"]
                lower = row["Lower Bound"]
                upper = row["Upper Bound"]

                if pd.isna(lower) or pd.isna(upper):
                    outlier_exprs.append(pl.lit(0).alias(col))
                else:
                    outlier_exprs.append(
                        ((pl.col(col) < lower) | (pl.col(col) > upper))
                        .sum()
                        .alias(col)
                    )

            outlier_counts = num_df.select(outlier_exprs).to_dicts()[0]
            summary_df["Outliers"] = summary_df["Column"].map(outlier_counts)
            summary_df["Outlier %"] = summary_df["Outliers"] / total_rows * 100

            summary_df["Distribution Shape"] = summary_df["Skewness"].apply(
                self._skew_label
            )

        summary_df["Status"] = summary_df.apply(
            lambda row: self._summary_status(row, mode),
            axis=1,
        )

        number_cols = summary_df.select_dtypes(include="number").columns
        summary_df[number_cols] = summary_df[number_cols].round(round_digits)

        total_missing = int(summary_df["Missing"].sum())
        zero_variance_cols = int(summary_df["Zero Variance"].sum())
        issue_cols = int((summary_df["Status"] != "Good").sum())

        cards = {
            "Quantitative Columns": len(numeric_cols),
            "Missing Numeric Values": total_missing,
            "Zero Variance Columns": zero_variance_cols,
            "Columns With Issues": issue_cols,
        }

        report_html = self._build_report_html(
            title="Quantitative Summary Report",
            subtitle=f"Mode: {mode.upper()}",
            cards=cards,
            sections={
                "📊 Quantitative Summary": summary_df
            },
        )

        self._display_html_report(report_html, file_name)

        return summary_df if return_data else None

    # ---------------------------------------------------------
    # Integrity report
    # ---------------------------------------------------------

    def integrity(
        self,
        columns=None,
        rules: dict | None = None,
        file_name: str = "integrity_report.html",
        max_examples: int = 5,
        return_data: bool = True,
    ):
        self._validate_filename(file_name)
        self._validate_positive_int(max_examples, "max_examples")

        selected_cols = self._resolve_columns(columns)
        df_pl = self._to_polars(self.dataframe).select(selected_cols)

        total_rows = df_pl.height
        report_rows = []

        rules = rules or {}

        for col, dtype in df_pl.schema.items():
            col_rules = rules.get(col, {})
            issues = []

            null_count = df_pl.select(pl.col(col).is_null().sum()).item()
            unique_count = df_pl.select(pl.col(col).n_unique()).item()

            if col_rules.get("required", False) and null_count > 0:
                issues.append((f"{null_count} missing values", null_count))

            if col_rules.get("unique", False):
                duplicate_count = df_pl.select(pl.col(col).is_duplicated().sum()).item()
                if duplicate_count > 0:
                    issues.append((f"{duplicate_count} duplicate values", duplicate_count))

            if col_rules.get("dtype") is not None:
                expected = col_rules["dtype"]
                if not self._dtype_matches(dtype, expected):
                    issues.append((f"Expected dtype {expected}, found {dtype}", total_rows))

            if dtype.is_numeric():
                numeric_issues = self._check_numeric_rules(df_pl, col, col_rules)
                issues.extend(numeric_issues)

            else:
                text_issues = self._check_text_rules(df_pl, col, col_rules)
                issues.extend(text_issues)

            allowed = col_rules.get("allowed")
            if allowed is not None:
                invalid_count = df_pl.select(
                    (~pl.col(col).is_in(allowed) & pl.col(col).is_not_null()).sum()
                ).item()

                if invalid_count > 0:
                    issues.append((f"{invalid_count} values outside allowed set", invalid_count))

            regex = col_rules.get("regex")
            if regex is not None:
                invalid_count = df_pl.select(
                    (
                        pl.col(col)
                        .cast(pl.String)
                        .str.contains(regex)
                        .not_()
                        & pl.col(col).is_not_null()
                    ).sum()
                ).item()

                if invalid_count > 0:
                    issues.append((f"{invalid_count} values failed regex pattern", invalid_count))

            custom = col_rules.get("custom")
            if custom is not None:
                invalid_count = self._check_custom_rule(df_pl, col, custom)

                if invalid_count > 0:
                    issues.append((f"{invalid_count} values failed custom rule", invalid_count))

            issue_count = sum(count for _, count in issues)

            if not issues:
                status = "Good"
            elif total_rows and issue_count / total_rows >= 0.10:
                status = "Needs Review"
            else:
                status = "Minor Issue"

            examples = (
                df_pl
                .filter(pl.col(col).is_not_null())
                .select(col)
                .head(max_examples)
                .to_series()
                .to_list()
            )

            report_rows.append({
                "Column": col,
                "Detected Type": str(dtype),
                "Rows": total_rows,
                "Missing": null_count,
                "Missing %": round((null_count / total_rows * 100), 2) if total_rows else 0,
                "Unique": unique_count,
                "Rules Applied": ", ".join(col_rules.keys()) if col_rules else "Default checks only",
                "Issues Found": "; ".join(msg for msg, _ in issues) if issues else "No major issues detected",
                "Status": status,
                "Examples": ", ".join(map(str, examples)),
            })

        report_df = pd.DataFrame(report_rows)

        cards = {
            "Columns Checked": len(report_df),
            "Good Columns": int((report_df["Status"] == "Good").sum()),
            "Columns With Issues": int((report_df["Status"] != "Good").sum()),
            "Needs Review": int((report_df["Status"] == "Needs Review").sum()),
        }

        report_html = self._build_report_html(
            title="Column Integrity Report",
            subtitle="Flexible rule-based data validation report",
            cards=cards,
            sections={"🧪 Column Integrity Summary": report_df},
        )

        self._display_html_report(report_html, file_name)

        return report_df if return_data else None


    def _check_numeric_rules(self, df_pl, col, rules):
        issues = []

        if rules.get("positive", False):
            count = df_pl.select(
                ((pl.col(col) <= 0) & pl.col(col).is_not_null()).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} non-positive values", count))

        if rules.get("non_negative", False):
            count = df_pl.select(
                ((pl.col(col) < 0) & pl.col(col).is_not_null()).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} negative values", count))

        min_value = rules.get("min")
        max_value = rules.get("max")

        if min_value is not None:
            count = df_pl.select(
                ((pl.col(col) < min_value) & pl.col(col).is_not_null()).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} values below minimum {min_value}", count))

        if max_value is not None:
            count = df_pl.select(
                ((pl.col(col) > max_value) & pl.col(col).is_not_null()).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} values above maximum {max_value}", count))

        return issues


    def _check_text_rules(self, df_pl, col, rules):
        issues = []

        s = pl.col(col).cast(pl.String)

        if rules.get("not_empty", False):
            count = df_pl.select(
                ((s.str.strip_chars().str.len_chars() == 0) & pl.col(col).is_not_null()).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} empty or whitespace-only values", count))

        if rules.get("isalpha", False):
            count = df_pl.select(
                (
                    s.str.contains(r"^[A-Za-z]+$").not_()
                    & pl.col(col).is_not_null()
                ).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} non-alpha values", count))

        if rules.get("isnumeric", False):
            count = df_pl.select(
                (
                    s.str.contains(r"^[0-9]+$").not_()
                    & pl.col(col).is_not_null()
                ).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} non-numeric text values", count))

        if rules.get("isalnum", False):
            count = df_pl.select(
                (
                    s.str.contains(r"^[A-Za-z0-9]+$").not_()
                    & pl.col(col).is_not_null()
                ).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} non-alphanumeric values", count))

        allowed_chars = rules.get("allowed_chars")

        if allowed_chars is not None:
            pattern = f"^[{allowed_chars}]+$"

            count = df_pl.select(
                (
                    s.str.contains(pattern).not_()
                    & pl.col(col).is_not_null()
                ).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} values contain disallowed characters", count))

        min_length = rules.get("min_length")
        max_length = rules.get("max_length")

        if min_length is not None:
            count = df_pl.select(
                (
                    (s.str.len_chars() < min_length)
                    & pl.col(col).is_not_null()
                ).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} values shorter than {min_length} characters", count))

        if max_length is not None:
            count = df_pl.select(
                (
                    (s.str.len_chars() > max_length)
                    & pl.col(col).is_not_null()
                ).sum()
            ).item()

            if count > 0:
                issues.append((f"{count} values longer than {max_length} characters", count))

        return issues


    def _check_custom_rule(self, df_pl, col, custom_func):
        values = df_pl.select(col).to_series().to_list()

        invalid_count = 0

        for value in values:
            if value is None:
                continue

            try:
                if not custom_func(value):
                    invalid_count += 1
            except Exception:
                invalid_count += 1

        return invalid_count


    @staticmethod
    def _dtype_matches(dtype, expected):
        expected = str(expected).lower()

        if expected in ["number", "numeric", "int", "float"]:
            return dtype.is_numeric()

        if expected in ["text", "string", "str"]:
            return dtype in [pl.String, pl.Categorical, pl.Enum]

        if expected in ["bool", "boolean"]:
            return dtype == pl.Boolean

        if expected in ["date", "datetime", "time"]:
            return dtype in [pl.Date, pl.Datetime, pl.Time]

        return str(dtype).lower() == expected

        # ---------------------------------------------------------
        # Data health report
        # ---------------------------------------------------------

    def data_health(
        self,
        title: str = "Data Health Overview",
        file_name: str = "health_report.html",
        show_problem_rows: bool = True,
        max_problem_rows: int | None = None,
    ) -> None:
        self._validate_filename(file_name)

        if max_problem_rows is not None:
            self._validate_positive_int(max_problem_rows, "max_problem_rows")

        df_pl = self._to_polars(self.dataframe).with_row_index("index")

        total_rows = df_pl.height
        data_cols = [col for col in df_pl.columns if col != "index"]
        total_cols = len(data_cols)
        total_cells = total_rows * total_cols

        missing_counts = df_pl.select([
            pl.col(col).is_null().sum().alias(col)
            for col in data_cols
        ])

        missing_col_df_pl = (
            missing_counts
            .transpose(
                include_header=True,
                header_name="Column",
                column_names=["Missing Count"],
            )
            .filter(pl.col("Missing Count") > 0)
            .with_columns(
                ((pl.col("Missing Count") / total_rows) * 100)
                .round(2)
                .alias("Missing Percent")
            )
            .with_columns(
                pl.when(pl.col("Missing Percent") >= 10)
                .then(pl.lit("Needs Review"))
                .when(pl.col("Missing Percent") >= 5)
                .then(pl.lit("Moderate Issue"))
                .otherwise(pl.lit("Low Issue"))
                .alias("Status")
            )
        )

        total_missing = int(missing_counts.sum_horizontal().item())
        missing_percent_total = (total_missing / total_cells * 100) if total_cells else 0

        missing_row_expr = pl.any_horizontal([
            pl.col(col).is_null()
            for col in data_cols
        ])

        duplicate_expr = pl.struct(data_cols).is_duplicated()

        df_issues = df_pl.with_columns([
            missing_row_expr.alias("_has_missing"),
            duplicate_expr.alias("_is_duplicate"),
            pl.sum_horizontal([
                pl.col(col).is_null().cast(pl.Int64)
                for col in data_cols
            ]).alias("_missing_count"),
        ])

        problem_rows_pl = df_issues.filter(
            pl.col("_has_missing") | pl.col("_is_duplicate")
        )

        duplicate_rows_pl = df_issues.filter(pl.col("_is_duplicate"))
        duplicate_count = duplicate_rows_pl.height

        missing_columns_expr = pl.concat_str([
            pl.when(pl.col(col).is_null())
            .then(pl.lit(col + ", "))
            .otherwise(pl.lit(""))
            for col in data_cols
        ]).str.strip_chars(", ")

        missing_row_df_pl = (
            df_issues
            .filter(pl.col("_has_missing"))
            .select([
                pl.col("index").alias("Row Index"),
                missing_columns_expr.alias("Missing Columns"),
                pl.col("_missing_count").alias("Missing Count"),
            ])
        )

        duplicate_row_df_pl = (
            duplicate_rows_pl
            .select([
                pl.col("index").alias("Row Index"),
                pl.lit("Duplicate Record").alias("Status"),
            ])
        )

        issue_type_expr = (
            pl.when(pl.col("_has_missing") & pl.col("_is_duplicate"))
            .then(pl.concat_str([
                pl.lit("Missing: "),
                missing_columns_expr,
                pl.lit(" | Duplicate row"),
            ]))
            .when(pl.col("_has_missing"))
            .then(pl.concat_str([
                pl.lit("Missing: "),
                missing_columns_expr,
            ]))
            .when(pl.col("_is_duplicate"))
            .then(pl.lit("Duplicate row"))
            .otherwise(pl.lit(""))
        )

        severity_expr = (
            pl.when(pl.col("_is_duplicate") | (pl.col("_missing_count") >= 3))
            .then(pl.lit("High"))
            .when(pl.col("_missing_count") == 2)
            .then(pl.lit("Medium"))
            .otherwise(pl.lit("Low"))
        )

        attention_df_pl = (
            problem_rows_pl
            .with_columns([
                issue_type_expr.alias("Issue Type"),
                severity_expr.alias("Severity"),
            ])
            .select([
                pl.col("index").alias("Row Index"),
                "Issue Type",
                "Severity",
            ])
        )

        problem_rows_display_pl = (
            problem_rows_pl
            .with_columns([
                issue_type_expr.alias("Issue Details"),
                severity_expr.alias("Severity"),
            ])
            .select(["Issue Details", "Severity", "index"] + data_cols)
        )

        if max_problem_rows is not None:
            problem_rows_display_pl = problem_rows_display_pl.head(max_problem_rows)

        completeness_score = 100 - missing_percent_total
        duplicate_score = 100 - ((duplicate_count / total_rows) * 100 if total_rows else 0)
        health_score = round((0.7 * completeness_score) + (0.3 * duplicate_score))

        missing_col_df = missing_col_df_pl.to_pandas()
        missing_row_df = missing_row_df_pl.to_pandas()
        duplicate_row_df = duplicate_row_df_pl.to_pandas()
        attention_df = attention_df_pl.to_pandas()
        self.problem_rows = problem_rows_display_pl.to_pandas()

        cards = {
            "Dataset Health Score": f"{health_score} / 100",
            "Missing Cells Found": total_missing,
            "Columns With Missing Values": len(missing_col_df),
            "Duplicate Rows Found": duplicate_count,
        }

        report_html = self._build_report_html(
            title=title,
            subtitle="Focused report: health score, missing values, duplicate rows, and affected records",
            cards=cards,
            sections={
                "⚠️ Missing Values by Column": missing_col_df,
                "📍 Rows With Missing Values": missing_row_df,
                "🔁 Duplicate Rows Found": duplicate_row_df,
                "🧾 Rows That Need Attention": attention_df,
            },
        )

        self._display_html_report(report_html, file_name)

        if show_problem_rows:
            if self.problem_rows.empty:
                display(HTML("<p>No problem rows found.</p>"))
            else:
                self.display(
                    dataframe=self.problem_rows,
                    caption="Filtered Table: Rows With Problems",
                )

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _get_title_xanchor(self) -> str:
        if self.title_align <= 0.3:
            return "left"
        if self.title_align >= 0.7:
            return "right"
        return "center"

    def _make_row_colors(self, row_count: int) -> list[str]:
        colors = self.cell_fillcolor

        if isinstance(colors, str):
            return [colors] * row_count

        repeated = colors * (row_count // len(colors) + 1)
        return repeated[:row_count]

    @staticmethod
    def _format_values(values, bold: bool = False, italic: bool = False) -> list[str]:
        formatted = [html.escape(str(value)) for value in values]

        if bold:
            formatted = [f"<b>{value}</b>" for value in formatted]

        if italic:
            formatted = [f"<i>{value}</i>" for value in formatted]

        return formatted

    @staticmethod
    def _to_polars(dataframe: Any) -> pl.DataFrame:
        if isinstance(dataframe, pl.DataFrame):
            return dataframe.clone()

        if isinstance(dataframe, pd.DataFrame):
            return pl.from_pandas(dataframe)

        return pl.from_pandas(Table.to_pandas(dataframe))

    @staticmethod
    def to_pandas(dataframe: Any) -> pd.DataFrame:
        if isinstance(dataframe, pd.DataFrame):
            return dataframe

        module = dataframe.__class__.__module__

        if module.startswith("polars"):
            return dataframe.to_pandas()

        if module.startswith("dask"):
            return dataframe.compute()

        if module.startswith("pyspark"):
            return dataframe.toPandas()

        if module.startswith("cudf"):
            return dataframe.to_pandas()

        if module.startswith("vaex"):
            return dataframe.to_pandas_df()

        raise TypeError(f"Unsupported dataframe type: {type(dataframe)}")

    @staticmethod
    def _skew_label(x) -> str:
        if pd.isna(x):
            return "Unknown"
        if x >= 1:
            return "Highly Right-Skewed"
        if x >= 0.5:
            return "Moderately Right-Skewed"
        if x <= -1:
            return "Highly Left-Skewed"
        if x <= -0.5:
            return "Moderately Left-Skewed"
        return "Approximately Symmetric"

    @staticmethod
    def _summary_status(row: pd.Series, mode: str) -> str:
        if row["Zero Variance"]:
            return "Needs Review"

        if row["Missing %"] >= 10:
            return "Needs Review"

        if mode == "full" and "Outlier %" in row and row["Outlier %"] >= 10:
            return "Needs Review"

        if row["Missing %"] > 0:
            return "Minor Issue"

        if mode == "full" and "Outlier %" in row and row["Outlier %"] > 0:
            return "Minor Issue"

        return "Good"

    @staticmethod
    def _issue_status(issue_count: int, total_rows: int, issues: list[str]) -> str:
        if not issues:
            return "Good"

        if total_rows and issue_count / total_rows >= 0.10:
            return "Needs Review"

        return "Minor Issue"

    @staticmethod
    def _status_class(value: str) -> str:
        if value in ["Needs Review", "High", "Duplicate Record"]:
            return "bad-row"

        if value in ["Minor Issue", "Medium", "Moderate Issue", "Low Issue"]:
            return "issue-row"

        return ""

    @classmethod
    def _df_to_html_table(cls, data: pd.DataFrame) -> str:
        if data.empty:
            return "<p>No issues found.</p>"

        output = "<table><tr>"

        for col in data.columns:
            output += f"<th>{html.escape(str(col))}</th>"

        output += "</tr>"

        for _, row in data.iterrows():
            row_class = ""

            if "Severity" in data.columns:
                row_class = cls._status_class(str(row["Severity"]))
            elif "Status" in data.columns:
                row_class = cls._status_class(str(row["Status"]))

            output += f"<tr class='{row_class}'>"

            for value in row:
                output += f"<td>{html.escape(str(value))}</td>"

            output += "</tr>"

        output += "</table>"
        return output

    @classmethod
    def _build_report_html(
        cls,
        title: str,
        subtitle: str,
        cards: dict[str, Any],
        sections: dict[str, pd.DataFrame],
    ) -> str:
        css = """
        <style>
        .dakit-report {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: auto;
        }

        .report-title {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .report-subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
        }

        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }

        .card {
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 20px;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,.08);
            text-align: center;
        }

        .card-value {
            font-size: 30px;
            font-weight: bold;
            color: #263238;
        }

        details {
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 12px;
            background: #fafafa;
            padding: 12px;
        }

        summary {
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            background: white;
            font-size: 13px;
        }

        th {
            background: #263238;
            color: white;
            padding: 9px;
            text-align: left;
        }

        td {
            padding: 9px;
            border-bottom: 1px solid #ddd;
            vertical-align: top;
        }

        .issue-row {
            background: #fff3e0;
        }

        .bad-row {
            background: #ffebee;
        }

        .download-btn {
            display: inline-block;
            padding: 12px 18px;
            background: #263238;
            color: white !important;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            margin: 20px 0;
        }
        </style>
        """

        card_html = ""

        for label, value in cards.items():
            card_html += f"""
            <div class="card">
                <div class="card-value">{html.escape(str(value))}</div>
                <div>{html.escape(str(label))}</div>
            </div>
            """

        section_html = ""

        for section_title, df in sections.items():
            section_html += f"""
            <details open>
                <summary>{html.escape(section_title)}</summary>
                {cls._df_to_html_table(df)}
            </details>
            """

        return f"""
        {css}
        <div class="dakit-report">
            <div class="report-title">{html.escape(title)}</div>
            <div class="report-subtitle">{html.escape(subtitle)}</div>

            <div class="card-grid">
                {card_html}
            </div>

            {section_html}
        </div>
        """

    def _resolve_columns(self, columns=None):
        if columns is None:
            return list(self.dataframe.columns)

        if not isinstance(columns, list):
            raise TypeError("columns must be a list of column names or index numbers.")

        resolved = []

        for col in columns:
            if isinstance(col, int):
                if col < 0 or col >= len(self.dataframe.columns):
                    raise IndexError(f"Column index {col} is out of range.")
                resolved.append(self.dataframe.columns[col])

            elif isinstance(col, str):
                if col not in self.dataframe.columns:
                    raise ValueError(f"Column '{col}' was not found in the dataframe.")
                resolved.append(col)

            else:
                raise TypeError("Each column must be a string name or integer index.")

        return resolved

    @staticmethod
    def _display_html_report(report_html: str, file_name: str) -> None:
        encoded = base64.b64encode(report_html.encode()).decode()

        download_button = f"""
        <div style="text-align:center;">
            <a class="download-btn" download="{html.escape(file_name)}" href="data:text/html;base64,{encoded}">
                Download HTML Report
            </a>
        </div>
        """

        display(HTML(report_html + download_button))

    # ---------------------------------------------------------
    # Validators
    # ---------------------------------------------------------

    @staticmethod
    def _validate_header_names(header_names: list[str], dataframe: pd.DataFrame) -> None:
        if not isinstance(header_names, list):
            raise TypeError("header_names must be a list of strings.")

        if len(header_names) != len(dataframe.columns):
            raise ValueError("header_names must match the number of dataframe columns.")

        if not all(isinstance(name, str) for name in header_names):
            raise TypeError("All header_names must be strings.")

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")

    @staticmethod
    def _validate_align(value: str, name: str = "align") -> None:
        valid = ["left", "center", "right"]

        if value not in valid:
            raise ValueError(f"{name} must be one of {valid}.")

    @staticmethod
    def _validate_margin(margin: dict[str, int]) -> None:
        if not isinstance(margin, dict):
            raise TypeError("margin must be a dictionary.")

        required_keys = {"l", "r", "t", "b"}

        if not required_keys.issubset(margin):
            raise ValueError("margin must contain l, r, t, and b.")

        for key, value in margin.items():
            if not isinstance(value, int):
                raise TypeError(f"margin['{key}'] must be an integer.")

            if value < 0:
                raise ValueError(f"margin['{key}'] cannot be negative.")

    @staticmethod
    def _validate_color(value: str, name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string color value.")

        if not value.strip():
            raise ValueError(f"{name} cannot be empty.")

    @staticmethod
    def _validate_color_or_list(value: str | list[str], name: str) -> None:
        if isinstance(value, str):
            Table._validate_color(value, name)
            return

        if isinstance(value, list):
            if not value:
                raise ValueError(f"{name} list cannot be empty.")

            for color in value:
                Table._validate_color(color, name)

            return

        raise TypeError(f"{name} must be a string or a list of strings.")

    @staticmethod
    def _validate_font_family(value: str, name: str = "font_family") -> None:
        valid = [
            "Arial",
            "Calibri",
            "Helvetica",
            "Times New Roman",
            "Courier New",
            "Verdana",
        ]

        if value not in valid:
            raise ValueError(f"{name} must be one of {valid}.")

    @staticmethod
    def _validate_bool(value: bool, name: str) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be True or False.")

    @staticmethod
    def _validate_filename(filename: str) -> None:
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")

        if not filename.strip():
            raise ValueError("filename cannot be empty.")

    @staticmethod
    def _validate_column_widths(column_widths: list[int]) -> None:
        if not isinstance(column_widths, list):
            raise TypeError("column_widths must be a list of integers.")

        if not column_widths:
            raise ValueError("column_widths cannot be empty.")

        for width in column_widths:
            if not isinstance(width, int):
                raise TypeError("Each column width must be an integer.")

            if width <= 0:
                raise ValueError("Each column width must be greater than 0.")