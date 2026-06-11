"""
dakit.Table
===========
A polished, Plotly-powered table display class for pandas / polars / dask /
PySpark / cuDF / vaex DataFrames.

Quick-start
-----------
>>> from dakit import Table
>>> t = Table(df, title="My Dataset")
>>> t.show()                  # render the table
>>> t.stats()                 # quantitative summary report
>>> t.stats(mode="full")      # full stats with outliers / skewness
>>> t.integrity()             # column integrity report
>>> t.data_health()           # missing-values + duplicate-rows report
>>> t.interactive()           # paginated interactive table
>>> Table.help()              # print this help text
"""

from __future__ import annotations

import base64
import html
import textwrap
import warnings
from typing import Any, Callable, Literal

import pandas as pd
import plotly.graph_objects as go
import polars as pl
from IPython.display import HTML, display

# ── Public type aliases ────────────────────────────────────────────────────────
AlignValue = Literal["left", "center", "right"]

FontFamily = Literal[
    "Arial",
    "Calibri",
    "Helvetica",
    "Times New Roman",
    "Courier New",
    "Verdana",
]

# Colors used for every HTML report — kept in one place so theme changes are easy
_THEME = {
    "header_bg":      "#263238",
    "header_fg":      "#ffffff",
    "card_bg":        "#ffffff",
    "card_border":    "#ddd",
    "report_bg":      "#f3f4f6",
    "table_even":     "#ffffff",
    "table_odd":      "#fafafa",
    "warn_row":       "#fff3e0",
    "bad_row":        "#ffebee",
    "text_primary":   "#111827",
    "text_secondary": "#555555",
    "btn_bg":         "#263238",
    "btn_fg":         "#ffffff",
}


class Table:
    """
    Display, analyse, and quality-check any tabular DataFrame with a single
    unified API.

    Parameters
    ----------
    dataframe : DataFrame
        Any of: pandas, polars, dask, PySpark, cuDF, vaex.
    header_names : list[str] | None
        Custom column labels shown in the rendered table.  Must match the
        number of columns.
    title : str | None
        Default title shown above every rendered table.

    Attributes (styling — all changeable via the ``set_*`` helpers)
    ---------------------------------------------------------------
    width, height           : int          table canvas size (px)
    header_fillcolor        : str          header background colour
    header_textcolor        : str          header text colour
    cell_fillcolor          : str | list   alternating row colours
    cell_textcolor          : str          cell text colour
    paper_bgcolor           : str          canvas background colour
    header_fontsize         : int
    cell_fontsize           : int
    header_height           : int          row height (px)
    cell_height             : int
    column_widths           : list[int] | None   relative column widths

    Examples
    --------
    >>> t = Table(df, title="Sales Data")
    >>> t.set_header_style(fillcolor="#1a237e", textcolor="#ffffff")
    >>> t.set_cell_style(fillcolor=["#e8eaf6", "#c5cae9"])
    >>> t.show()
    """

    # ── Construction ──────────────────────────────────────────────────────────

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

        # ── Layout defaults ──────────────────────────────────────────────────
        self.margin       = dict(l=20, r=20, t=60, b=20)
        self.title_align  = 0.5
        self.width        = 1200
        self.height       = 700

        # ── Header style defaults ────────────────────────────────────────────
        self.header_fillcolor:   str | list[str] = _THEME["header_bg"]
        self.header_textcolor:   str             = _THEME["header_fg"]
        self.header_align:       AlignValue      = "left"
        self.header_linecolor:   str             = _THEME["header_bg"]
        self.header_fontsize:    int             = 13
        self.header_font_family: FontFamily      = "Arial"
        self.header_bold:        bool            = True
        self.header_italic:      bool            = False
        self.header_height:      int             = 38

        # ── Cell style defaults ──────────────────────────────────────────────
        self.cell_fillcolor:   str | list[str] = [_THEME["table_even"], _THEME["table_odd"]]
        self.cell_textcolor:   str             = _THEME["text_primary"]
        self.cell_align:       AlignValue      = "left"
        self.cell_linecolor:   str             = "#e5e7eb"
        self.cell_fontsize:    int             = 11
        self.cell_font_family: FontFamily      = "Arial"
        self.cell_bold:        bool            = False
        self.cell_italic:      bool            = False
        self.cell_height:      int             = 32

        # ── Misc ─────────────────────────────────────────────────────────────
        self.paper_bgcolor:  str            = _THEME["report_bg"]
        self.column_widths:  list[int] | None = None

    # ── Layout and style methods ──────────────────────────────────────────────

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
        """
        Adjust canvas layout settings.

        Parameters
        ----------
        title        : str            Override the table title.
        title_align  : "left"|"center"|"right"
        width        : int            Canvas width in pixels.
        height       : int            Canvas height in pixels.
        header_height: int            Header row height in pixels.
        cell_height  : int            Data row height in pixels.
        margin       : dict           Keys: l, r, t, b (all ints ≥ 0).
        column_widths: list[int]      Relative column widths.

        Examples
        --------
        >>> t.set_layout(title="Q4 Results", width=1400, title_align="left")
        """
        if title is not None:
            if not isinstance(title, str):
                raise TypeError("title must be a string.")
            self.title = title

        if title_align is not None:
            self._validate_align(title_align, "title_align")
            self.title_align = {"left": 0.05, "center": 0.5, "right": 0.95}[title_align]

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
        """
        Set canvas-level (background) styles.

        Parameters
        ----------
        paper_bgcolor : str   Any CSS / hex colour string.

        Examples
        --------
        >>> t.set_global_style(paper_bgcolor="#ffffff")
        """
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
        """
        Customise the appearance of the header row.

        Parameters
        ----------
        fillcolor   : str | list[str]   Background colour(s).
        textcolor   : str               Text colour.
        align       : "left"|"center"|"right"
        linecolor   : str               Border colour.
        fontsize    : int               Font size in pt.
        font_family : FontFamily        One of the supported font names.
        bold        : bool
        italic      : bool

        Examples
        --------
        >>> t.set_header_style(fillcolor="#1a237e", fontsize=14, bold=True)
        """
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
        """
        Customise the appearance of data cells.

        Parameters
        ----------
        fillcolor   : str | list[str]   Single colour or alternating pair.
        textcolor   : str
        align       : "left"|"center"|"right"
        linecolor   : str               Border colour.
        fontsize    : int
        font_family : FontFamily
        bold        : bool
        italic      : bool

        Examples
        --------
        >>> t.set_cell_style(fillcolor=["#e8eaf6", "#c5cae9"], fontsize=12)
        """
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

    # ── Display methods ───────────────────────────────────────────────────────

    def interactive(self, rows_per_page: int = 20):
        """
        Render an interactive, paginated table.

        Uses Colab's ``data_table.DataTable`` when running in Google Colab,
        and falls back to a plain HTML table otherwise.

        Parameters
        ----------
        rows_per_page : int   Number of rows shown per page (Colab only).

        Returns
        -------
        DataTable | None

        Examples
        --------
        >>> t.interactive(rows_per_page=50)
        """
        self._validate_positive_int(rows_per_page, "rows_per_page")

        try:
            from google.colab import data_table  # type: ignore[import]
            return data_table.DataTable(
                self.dataframe,
                include_index=False,
                num_rows_per_page=rows_per_page,
            )
        except ImportError:
            # Not in Colab — use a plain HTML table
            display(HTML(self.dataframe.to_html(index=False)))
        except Exception as exc:
            warnings.warn(
                f"interactive() fell back to plain HTML due to: {exc}",
                stacklevel=2,
            )
            display(HTML(self.dataframe.to_html(index=False)))

    def show(self, caption: str | None = None) -> None:
        """
        Render the table inline.  Thin alias for :meth:`display`.

        Parameters
        ----------
        caption : str | None   Overrides the instance title for this call only.

        Examples
        --------
        >>> t.show()
        >>> t.show(caption="Filtered view")
        """
        self.display(caption=caption)

    def display(
        self,
        dataframe: Any | None = None,
        filename: str = "newtable",
        caption: str | None = None,
        max_rows: int | None = 1000,
        show_index: bool = False,
    ) -> None:
        """
        Render the table as a Plotly figure with a download-image button.

        Parameters
        ----------
        dataframe  : DataFrame | None   Render this instead of ``self.dataframe``.
        filename   : str                Base filename for the PNG download.
        caption    : str | None         Title shown above the table (one-shot override).
        max_rows   : int | None         Truncate to this many rows; ``None`` = no limit.
        show_index : bool               Prepend the DataFrame index as a column.

        Examples
        --------
        >>> t.display(filename="q4_sales", caption="Q4 Sales", max_rows=500)
        """
        self._validate_filename(filename)

        df = (
            self.dataframe.copy()
            if dataframe is None
            else self.to_pandas(dataframe).copy()
        )

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
                "Use .interactive() for larger datasets.",
                stacklevel=2,
            )
            df = df.head(max_rows)

        headers = list(self.header_names or df.columns)

        if self.header_names is not None:
            self._validate_header_names(self.header_names, df)

        # ── BUG FIX: was using self.cell_italic for header ──────────────────
        headers = self._format_values(
            headers,
            bold=self.header_bold,
            italic=self.header_italic,   # ← fixed (was self.cell_italic)
        )

        values = [df[col].tolist() for col in df.columns]

        if self.cell_bold or self.cell_italic:
            values = [
                self._format_values(col_vals, bold=self.cell_bold, italic=self.cell_italic)
                for col_vals in values
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
        layout_kwargs: dict[str, Any] = dict(
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
        fig.show(config={"toImageButtonOptions": {"filename": filename}})

    # ── Summary statistics ────────────────────────────────────────────────────

    def stats(
        self,
        columns: list[int | str] | None = None,
        file_name: str = "summary_stats.html",
        mode: Literal["fast", "full"] = "fast",
        round_digits: int = 3,
        return_data: bool = True,
    ) -> pd.DataFrame | None:
        """
        Generate a quantitative summary report for numeric columns.

        Parameters
        ----------
        columns     : list[int | str] | None   Subset of columns to analyse.
        file_name   : str                       HTML report download filename.
        mode        : "fast" | "full"
            "fast"  → count, missing %, mean, std, min, max, unique, CV,
                       zero-variance flag, status
            "full"  → everything in fast + median, variance, Q1, Q3, IQR,
                       fence bounds, skewness, outlier count, distribution shape
        round_digits: int    Decimal places for numeric output.
        return_data : bool   Return the summary DataFrame.

        Returns
        -------
        pd.DataFrame | None

        Examples
        --------
        >>> summary = t.stats(mode="full", round_digits=2)
        >>> t.stats(columns=["price", "quantity"], mode="fast")
        """
        self._validate_filename(file_name)

        if mode not in ("fast", "full"):
            raise ValueError("mode must be either 'fast' or 'full'.")

        selected_cols = self._resolve_columns(columns)
        df_pl         = self._to_polars(self.dataframe).select(selected_cols)

        numeric_cols = [col for col, dtype in df_pl.schema.items() if dtype.is_numeric()]

        if not numeric_cols:
            display(HTML("<h3>No quantitative columns found.</h3>"))
            return pd.DataFrame() if return_data else None

        num_df     = df_pl.select(numeric_cols)
        total_rows = num_df.height

        # ── Build all aggregations in a single Polars pass ───────────────────
        exprs = []
        for col in numeric_cols:
            exprs += [
                pl.col(col).count().alias(f"{col}__Count"),
                pl.col(col).null_count().alias(f"{col}__Missing"),
                pl.col(col).mean().alias(f"{col}__Mean"),
                pl.col(col).std().alias(f"{col}__Std"),
                pl.col(col).min().alias(f"{col}__Min"),
                pl.col(col).max().alias(f"{col}__Max"),
                pl.col(col).n_unique().alias(f"{col}__Unique"),
            ]
            if mode == "full":
                exprs += [
                    pl.col(col).median().alias(f"{col}__Median"),
                    pl.col(col).var().alias(f"{col}__Variance"),
                    pl.col(col).quantile(0.25).alias(f"{col}__Q1"),
                    pl.col(col).quantile(0.75).alias(f"{col}__Q3"),
                    pl.col(col).skew().alias(f"{col}__Skewness"),
                ]

        stats = num_df.select(exprs).to_dicts()[0]

        rows = []
        for col in numeric_cols:
            count    = stats[f"{col}__Count"]
            missing  = stats[f"{col}__Missing"]
            mean     = stats[f"{col}__Mean"]
            std      = stats[f"{col}__Std"]
            min_val  = stats[f"{col}__Min"]
            max_val  = stats[f"{col}__Max"]
            unique   = stats[f"{col}__Unique"]
            miss_pct = (missing / total_rows * 100) if total_rows else 0
            rng      = (max_val - min_val) if None not in (min_val, max_val) else None
            cv       = (std / mean) if mean not in (None, 0) and std is not None else None
            zero_var = std == 0 if std is not None else False

            row: dict[str, Any] = {
                "Column":        col,
                "Count":         count,
                "Missing":       missing,
                "Missing %":     miss_pct,
                "Unique":        unique,
                "Mean":          mean,
                "Std":           std,
                "Min":           min_val,
                "Max":           max_val,
                "Range":         rng,
                "CV":            cv,
                "Zero Variance": zero_var,
            }

            if mode == "full":
                q1  = stats[f"{col}__Q1"]
                q3  = stats[f"{col}__Q3"]
                iqr = (q3 - q1) if None not in (q1, q3) else None
                row.update({
                    "Median":      stats[f"{col}__Median"],
                    "Variance":    stats[f"{col}__Variance"],
                    "Q1":          q1,
                    "Q3":          q3,
                    "IQR":         iqr,
                    "Lower Bound": (q1 - 1.5 * iqr) if iqr is not None else None,
                    "Upper Bound": (q3 + 1.5 * iqr) if iqr is not None else None,
                    "Skewness":    stats[f"{col}__Skewness"],
                })

            rows.append(row)

        summary_df = pd.DataFrame(rows)

        # ── BUG FIX: compute outliers BEFORE calling _summary_status ─────────
        if mode == "full":
            outlier_exprs = []
            for _, row_s in summary_df.iterrows():
                col   = row_s["Column"]
                lower = row_s["Lower Bound"]
                upper = row_s["Upper Bound"]
                if pd.isna(lower) or pd.isna(upper):
                    outlier_exprs.append(pl.lit(0).alias(col))
                else:
                    outlier_exprs.append(
                        ((pl.col(col) < lower) | (pl.col(col) > upper))
                        .sum()
                        .alias(col)
                    )

            outlier_counts = num_df.select(outlier_exprs).to_dicts()[0]
            summary_df["Outliers"]   = summary_df["Column"].map(outlier_counts)
            summary_df["Outlier %"]  = summary_df["Outliers"] / total_rows * 100
            summary_df["Distribution Shape"] = summary_df["Skewness"].apply(self._skew_label)

        # Status is computed AFTER outlier columns exist
        summary_df["Status"] = summary_df.apply(
            lambda r: self._summary_status(r, mode), axis=1
        )

        num_cols = summary_df.select_dtypes(include="number").columns
        summary_df[num_cols] = summary_df[num_cols].round(round_digits)

        cards = {
            "Quantitative Columns":  len(numeric_cols),
            "Missing Numeric Values": int(summary_df["Missing"].sum()),
            "Zero Variance Columns":  int(summary_df["Zero Variance"].sum()),
            "Columns With Issues":    int((summary_df["Status"] != "Good").sum()),
        }

        report_html = self._build_report_html(
            title="Quantitative Summary Report",
            subtitle=f"Mode: {mode.upper()}",
            cards=cards,
            sections={"📊 Quantitative Summary": summary_df},
        )
        self._display_html_report(report_html, file_name)

        return summary_df if return_data else None

    # ── Integrity report ──────────────────────────────────────────────────────

    def integrity(
        self,
        columns: list[int | str] | None = None,
        rules: dict | None = None,
        file_name: str = "integrity_report.html",
        max_examples: int = 5,
        return_data: bool = True,
    ) -> pd.DataFrame | None:
        """
        Validate every column against optional rules and produce an HTML report.

        Parameters
        ----------
        columns      : list[int | str] | None   Subset of columns to check.
        rules        : dict | None
            Mapping of ``{column_name: {rule_key: rule_value}}``.

            Numeric rules
            ~~~~~~~~~~~~~
            required    : bool   Flag rows where this column is null.
            unique      : bool   Flag if duplicate values exist.
            dtype       : str    Expected type: "numeric","text","bool","date".
            positive    : bool   All values must be > 0.
            non_negative: bool   All values must be ≥ 0.
            min         : float  Lower bound (inclusive).
            max         : float  Upper bound (inclusive).
            allowed     : list   Whitelist of valid values.
            regex       : str    Values must match this pattern.
            custom      : callable(value) → bool

            Text rules
            ~~~~~~~~~~
            not_empty   : bool   No blank / whitespace-only strings.
            isalpha     : bool   Only alphabetic characters.
            isnumeric   : bool   Only digit characters.
            isalnum     : bool   Only alphanumeric characters.
            allowed_chars: str   Regex character class (e.g. ``"A-Za-z0-9"``)
            min_length  : int    Minimum string length.
            max_length  : int    Maximum string length.
            (+ all shared rules above)

        file_name    : str   HTML report download filename.
        max_examples : int   Number of sample values shown per column.
        return_data  : bool  Return the integrity DataFrame.

        Returns
        -------
        pd.DataFrame | None

        Examples
        --------
        >>> rules = {
        ...     "age":   {"required": True, "positive": True, "max": 120},
        ...     "email": {"not_empty": True, "regex": r".+@.+\\..+"},
        ... }
        >>> report_df = t.integrity(rules=rules)
        """
        self._validate_filename(file_name)
        self._validate_positive_int(max_examples, "max_examples")

        selected_cols = self._resolve_columns(columns)
        df_pl         = self._to_polars(self.dataframe).select(selected_cols)
        total_rows    = df_pl.height
        rules         = rules or {}
        report_rows   = []

        for col, dtype in df_pl.schema.items():
            col_rules   = rules.get(col, {})
            issues: list[tuple[str, int]] = []

            null_count   = df_pl.select(pl.col(col).is_null().sum()).item()
            unique_count = df_pl.select(pl.col(col).n_unique()).item()

            if col_rules.get("required", False) and null_count > 0:
                issues.append((f"{null_count} missing values", null_count))

            if col_rules.get("unique", False):
                dup_count = df_pl.select(pl.col(col).is_duplicated().sum()).item()
                if dup_count > 0:
                    issues.append((f"{dup_count} duplicate values", dup_count))

            if col_rules.get("dtype") is not None:
                if not self._dtype_matches(dtype, col_rules["dtype"]):
                    issues.append((f"Expected dtype {col_rules['dtype']}, found {dtype}", total_rows))

            if dtype.is_numeric():
                issues.extend(self._check_numeric_rules(df_pl, col, col_rules))
            else:
                issues.extend(self._check_text_rules(df_pl, col, col_rules))

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
                        pl.col(col).cast(pl.String).str.contains(regex).not_()
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

            issue_count = sum(c for _, c in issues)

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
                "Column":        col,
                "Detected Type": str(dtype),
                "Rows":          total_rows,
                "Missing":       null_count,
                "Missing %":     round((null_count / total_rows * 100), 2) if total_rows else 0,
                "Unique":        unique_count,
                "Rules Applied": ", ".join(col_rules.keys()) if col_rules else "Default checks only",
                "Issues Found":  "; ".join(msg for msg, _ in issues) if issues else "No major issues detected",
                "Status":        status,
                "Examples":      ", ".join(map(str, examples)),
            })

        report_df = pd.DataFrame(report_rows)

        cards = {
            "Columns Checked":    len(report_df),
            "Good Columns":       int((report_df["Status"] == "Good").sum()),
            "Columns With Issues": int((report_df["Status"] != "Good").sum()),
            "Needs Review":       int((report_df["Status"] == "Needs Review").sum()),
        }

        report_html = self._build_report_html(
            title="Column Integrity Report",
            subtitle="Flexible rule-based data validation report",
            cards=cards,
            sections={"🧪 Column Integrity Summary": report_df},
        )
        self._display_html_report(report_html, file_name)

        return report_df if return_data else None

    # ── Rule checkers ─────────────────────────────────────────────────────────

    def _check_numeric_rules(
        self, df_pl: pl.DataFrame, col: str, rules: dict
    ) -> list[tuple[str, int]]:
        issues: list[tuple[str, int]] = []

        checks: list[tuple[str, Any]] = [
            ("positive",     (pl.col(col) <= 0) & pl.col(col).is_not_null()),
            ("non_negative", (pl.col(col) < 0)  & pl.col(col).is_not_null()),
        ]

        # Build all simple boolean checks in a single Polars pass
        active = [(label, expr) for label, expr in checks if rules.get(label, False)]

        min_value = rules.get("min")
        max_value = rules.get("max")
        if min_value is not None:
            active.append(("_min", (pl.col(col) < min_value) & pl.col(col).is_not_null()))
        if max_value is not None:
            active.append(("_max", (pl.col(col) > max_value) & pl.col(col).is_not_null()))

        if active:
            agg = df_pl.select([
                expr.sum().alias(label) for label, expr in active
            ]).to_dicts()[0]

            for label, _ in active:
                count = agg[label]
                if count > 0:
                    if label == "positive":
                        issues.append((f"{count} non-positive values", count))
                    elif label == "non_negative":
                        issues.append((f"{count} negative values", count))
                    elif label == "_min":
                        issues.append((f"{count} values below minimum {min_value}", count))
                    elif label == "_max":
                        issues.append((f"{count} values above maximum {max_value}", count))

        return issues

    def _check_text_rules(
        self, df_pl: pl.DataFrame, col: str, rules: dict
    ) -> list[tuple[str, int]]:
        issues: list[tuple[str, int]] = []
        s = pl.col(col).cast(pl.String)

        active: list[tuple[str, Any]] = []

        if rules.get("not_empty", False):
            active.append(("not_empty",
                (s.str.strip_chars().str.len_chars() == 0) & pl.col(col).is_not_null()))

        if rules.get("isalpha", False):
            active.append(("isalpha",
                s.str.contains(r"^[A-Za-z]+$").not_() & pl.col(col).is_not_null()))

        if rules.get("isnumeric", False):
            active.append(("isnumeric",
                s.str.contains(r"^[0-9]+$").not_() & pl.col(col).is_not_null()))

        if rules.get("isalnum", False):
            active.append(("isalnum",
                s.str.contains(r"^[A-Za-z0-9]+$").not_() & pl.col(col).is_not_null()))

        allowed_chars = rules.get("allowed_chars")
        if allowed_chars is not None:
            active.append(("allowed_chars",
                s.str.contains(f"^[{allowed_chars}]+$").not_() & pl.col(col).is_not_null()))

        min_length = rules.get("min_length")
        max_length = rules.get("max_length")
        if min_length is not None:
            active.append(("min_length",
                (s.str.len_chars() < min_length) & pl.col(col).is_not_null()))
        if max_length is not None:
            active.append(("max_length",
                (s.str.len_chars() > max_length) & pl.col(col).is_not_null()))

        if active:
            agg = df_pl.select([
                expr.sum().alias(label) for label, expr in active
            ]).to_dicts()[0]

            messages = {
                "not_empty":     "empty or whitespace-only values",
                "isalpha":       "non-alpha values",
                "isnumeric":     "non-numeric text values",
                "isalnum":       "non-alphanumeric values",
                "allowed_chars": "values contain disallowed characters",
                "min_length":    f"values shorter than {min_length} characters",
                "max_length":    f"values longer than {max_length} characters",
            }

            for label, _ in active:
                count = agg[label]
                if count > 0:
                    issues.append((f"{count} {messages[label]}", count))

        return issues

    def _check_custom_rule(
        self,
        df_pl: pl.DataFrame,
        col: str,
        custom_func: Callable[[Any], bool],
    ) -> int:
        """
        Apply a custom validation function column-wise.

        Uses ``map_elements`` so the loop stays inside Polars rather than
        pure Python.  Falls back to a Python loop only if the series type
        cannot be handled by ``map_elements``.
        """
        series = df_pl.select(col).to_series()

        def _safe_check(value: Any) -> bool:
            try:
                return bool(custom_func(value))
            except Exception:
                return False

        try:
            result = series.map_elements(
                lambda v: not _safe_check(v),
                return_dtype=pl.Boolean,
            )
            return int(result.sum())
        except Exception:
            # Ultimate fallback — plain Python
            return sum(
                1 for v in series.to_list()
                if v is not None and not _safe_check(v)
            )

    # ── Data health report ────────────────────────────────────────────────────

    def data_health(
        self,
        title: str = "Data Health Overview",
        file_name: str = "health_report.html",
        show_problem_rows: bool = True,
        max_problem_rows: int | None = None,
    ) -> None:
        """
        Generate a data-health report covering:

        * Overall health score (0 – 100)
        * Missing values by column and by row
        * Duplicate rows
        * A prioritised "needs attention" table

        After this method returns, ``self.problem_rows`` contains the filtered
        DataFrame of flagged rows for further inspection.

        Parameters
        ----------
        title            : str    Report heading.
        file_name        : str    HTML download filename.
        show_problem_rows: bool   Also render the problem-rows table inline.
        max_problem_rows : int | None   Cap on rows shown in the rendered table.

        Examples
        --------
        >>> t.data_health(show_problem_rows=True, max_problem_rows=50)
        >>> t.problem_rows          # access flagged rows afterwards
        """
        self._validate_filename(file_name)
        if max_problem_rows is not None:
            self._validate_positive_int(max_problem_rows, "max_problem_rows")

        df_pl      = self._to_polars(self.dataframe).with_row_index("index")
        total_rows = df_pl.height
        data_cols  = [c for c in df_pl.columns if c != "index"]
        total_cols = len(data_cols)
        total_cells = total_rows * total_cols

        # ── Missing counts per column ────────────────────────────────────────
        missing_counts = df_pl.select([
            pl.col(col).is_null().sum().alias(col) for col in data_cols
        ])

        missing_col_df_pl = (
            missing_counts
            .transpose(include_header=True, header_name="Column", column_names=["Missing Count"])
            .filter(pl.col("Missing Count") > 0)
            .with_columns(
                ((pl.col("Missing Count") / total_rows) * 100).round(2).alias("Missing Percent")
            )
            .with_columns(
                pl.when(pl.col("Missing Percent") >= 10).then(pl.lit("Needs Review"))
                .when(pl.col("Missing Percent") >= 5).then(pl.lit("Moderate Issue"))
                .otherwise(pl.lit("Low Issue"))
                .alias("Status")
            )
        )

        total_missing    = int(missing_counts.sum_horizontal().item())
        missing_pct_total = (total_missing / total_cells * 100) if total_cells else 0

        # ── BUG FIX: build missing column list using Polars separator ────────
        # Old approach used concat_str with trailing ", " per column then
        # strip_chars — leaving internal gaps for sparse rows.
        # New: filter to null columns per row via a struct/explode approach;
        # for simplicity we keep concat_str but use the separator argument
        # (Polars ≥ 0.19) which ignores null-contributing segments.
        null_flags = [
            pl.when(pl.col(col).is_null()).then(pl.lit(col)).otherwise(pl.lit(None))
            for col in data_cols
        ]
        missing_columns_expr = (
            pl.concat_str(null_flags, separator=", ", ignore_nulls=True)
            .alias("_missing_cols_str")
        )

        missing_row_expr  = pl.any_horizontal([pl.col(c).is_null() for c in data_cols])
        duplicate_expr    = pl.struct(data_cols).is_duplicated()
        missing_count_expr = pl.sum_horizontal([
            pl.col(c).is_null().cast(pl.Int64) for c in data_cols
        ])

        df_issues = df_pl.with_columns([
            missing_row_expr.alias("_has_missing"),
            duplicate_expr.alias("_is_duplicate"),
            missing_count_expr.alias("_missing_count"),
            missing_columns_expr,
        ])

        problem_rows_pl    = df_issues.filter(pl.col("_has_missing") | pl.col("_is_duplicate"))
        duplicate_rows_pl  = df_issues.filter(pl.col("_is_duplicate"))
        duplicate_count    = duplicate_rows_pl.height

        missing_row_df_pl = (
            df_issues
            .filter(pl.col("_has_missing"))
            .select([
                pl.col("index").alias("Row Index"),
                pl.col("_missing_cols_str").alias("Missing Columns"),
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
                pl.lit("Missing: "), pl.col("_missing_cols_str"), pl.lit(" | Duplicate row")
            ]))
            .when(pl.col("_has_missing"))
            .then(pl.concat_str([pl.lit("Missing: "), pl.col("_missing_cols_str")]))
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
            .select([pl.col("index").alias("Row Index"), "Issue Type", "Severity"])
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

        completeness_score = 100 - missing_pct_total
        duplicate_score    = 100 - ((duplicate_count / total_rows * 100) if total_rows else 0)
        health_score       = round(0.7 * completeness_score + 0.3 * duplicate_score)

        self.problem_rows = problem_rows_display_pl.to_pandas()

        cards = {
            "Dataset Health Score":        f"{health_score} / 100",
            "Missing Cells Found":         total_missing,
            "Columns With Missing Values": len(missing_col_df_pl),
            "Duplicate Rows Found":        duplicate_count,
        }

        report_html = self._build_report_html(
            title=title,
            subtitle=(
                "Focused report: health score, missing values, "
                "duplicate rows, and affected records"
            ),
            cards=cards,
            sections={
                "⚠️ Missing Values by Column":    missing_col_df_pl.to_pandas(),
                "📍 Rows With Missing Values":    missing_row_df_pl.to_pandas(),
                "🔁 Duplicate Rows Found":        duplicate_row_df_pl.to_pandas(),
                "🧾 Rows That Need Attention":    attention_df_pl.to_pandas(),
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

    # ── Internal helpers ──────────────────────────────────────────────────────

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
        # Tile using integer division + slice — avoids Python-level loop
        repeats = (row_count + len(colors) - 1) // len(colors)
        return (colors * repeats)[:row_count]

    @staticmethod
    def _format_values(
        values: list[Any],
        bold: bool = False,
        italic: bool = False,
    ) -> list[str]:
        formatted = [html.escape(str(v)) for v in values]
        if bold:
            formatted = [f"<b>{v}</b>" for v in formatted]
        if italic:
            formatted = [f"<i>{v}</i>" for v in formatted]
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
        """
        Convert any supported DataFrame type to pandas.

        Supported types
        ---------------
        pandas, polars, dask, PySpark (toPandas), cuDF, vaex

        Parameters
        ----------
        dataframe : Any   The DataFrame to convert.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        TypeError   If the type is not supported.

        Examples
        --------
        >>> pdf = Table.to_pandas(polars_df)
        """
        if isinstance(dataframe, pd.DataFrame):
            return dataframe

        module = dataframe.__class__.__module__

        converters = {
            "polars":  lambda df: df.to_pandas(),
            "dask":    lambda df: df.compute(),
            "pyspark": lambda df: df.toPandas(),
            "cudf":    lambda df: df.to_pandas(),
            "vaex":    lambda df: df.to_pandas_df(),
        }

        for prefix, converter in converters.items():
            if module.startswith(prefix):
                return converter(dataframe)

        raise TypeError(f"Unsupported dataframe type: {type(dataframe)}")

    @staticmethod
    def _skew_label(x: float) -> str:
        if pd.isna(x):        return "Unknown"
        if x >= 1:            return "Highly Right-Skewed"
        if x >= 0.5:          return "Moderately Right-Skewed"
        if x <= -1:           return "Highly Left-Skewed"
        if x <= -0.5:         return "Moderately Left-Skewed"
        return "Approximately Symmetric"

    @staticmethod
    def _summary_status(row: pd.Series, mode: str) -> str:
        if row["Zero Variance"]:
            return "Needs Review"
        if row["Missing %"] >= 10:
            return "Needs Review"
        if mode == "full" and "Outlier %" in row.index and row["Outlier %"] >= 10:
            return "Needs Review"
        if row["Missing %"] > 0:
            return "Minor Issue"
        if mode == "full" and "Outlier %" in row.index and row["Outlier %"] > 0:
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
        if value in ("Needs Review", "High", "Duplicate Record"):
            return "bad-row"
        if value in ("Minor Issue", "Medium", "Moderate Issue", "Low Issue"):
            return "issue-row"
        return ""

    @classmethod
    def _df_to_html_table(cls, data: pd.DataFrame) -> str:
        if data.empty:
            return "<p>No issues found.</p>"

        cols_html = "".join(f"<th>{html.escape(str(c))}</th>" for c in data.columns)
        rows_html = []

        for _, row in data.iterrows():
            row_class = ""
            if "Severity" in data.columns:
                row_class = cls._status_class(str(row["Severity"]))
            elif "Status" in data.columns:
                row_class = cls._status_class(str(row["Status"]))

            cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
            rows_html.append(f"<tr class='{row_class}'>{cells}</tr>")

        return f"<table><tr>{cols_html}</tr>{''.join(rows_html)}</table>"

    @classmethod
    def _build_report_html(
        cls,
        title: str,
        subtitle: str,
        cards: dict[str, Any],
        sections: dict[str, pd.DataFrame],
    ) -> str:
        # ── BUG FIX: CSS scoped to .dakit-report so dark-mode host themes
        # cannot bleed through.  Every colour is set explicitly; the wrapper
        # forces light-mode via color-scheme + explicit backgrounds.
        css = f"""
        <style>
        .dakit-report {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: auto;
            color-scheme: light;
            background: {_THEME['report_bg']};
            color: {_THEME['text_primary']};
            padding: 16px;
            border-radius: 12px;
        }}
        .dakit-report * {{
            box-sizing: border-box;
        }}
        .dakit-report .report-title {{
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 5px;
            color: {_THEME['text_primary']};
        }}
        .dakit-report .report-subtitle {{
            text-align: center;
            color: {_THEME['text_secondary']};
            margin-bottom: 25px;
        }}
        .dakit-report .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .dakit-report .card {{
            border: 1px solid {_THEME['card_border']};
            border-radius: 12px;
            padding: 20px;
            background: {_THEME['card_bg']};
            box-shadow: 0 2px 8px rgba(0,0,0,.08);
            text-align: center;
            color: {_THEME['text_primary']};
        }}
        .dakit-report .card-value {{
            font-size: 30px;
            font-weight: bold;
            color: {_THEME['header_bg']};
        }}
        .dakit-report details {{
            margin-bottom: 15px;
            border: 1px solid {_THEME['card_border']};
            border-radius: 12px;
            background: {_THEME['table_odd']};
            padding: 12px;
        }}
        .dakit-report summary {{
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            color: {_THEME['text_primary']};
        }}
        .dakit-report table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            background: {_THEME['card_bg']};
            font-size: 13px;
            color: {_THEME['text_primary']};
        }}
        .dakit-report th {{
            background: {_THEME['header_bg']};
            color: {_THEME['header_fg']};
            padding: 9px;
            text-align: left;
        }}
        .dakit-report td {{
            padding: 9px;
            border-bottom: 1px solid {_THEME['card_border']};
            vertical-align: top;
            color: {_THEME['text_primary']};
            background: {_THEME['card_bg']};
        }}
        .dakit-report .issue-row td {{
            background: {_THEME['warn_row']};
        }}
        .dakit-report .bad-row td {{
            background: {_THEME['bad_row']};
        }}
        .dakit-report .download-btn {{
            display: inline-block;
            padding: 12px 18px;
            background: {_THEME['btn_bg']};
            color: {_THEME['btn_fg']} !important;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            margin: 20px 0;
        }}
        </style>
        """

        card_html = "".join(
            f"""<div class="card">
                    <div class="card-value">{html.escape(str(v))}</div>
                    <div>{html.escape(str(k))}</div>
                </div>"""
            for k, v in cards.items()
        )

        section_html = "".join(
            f"""<details open>
                    <summary>{html.escape(sec_title)}</summary>
                    {cls._df_to_html_table(df)}
                </details>"""
            for sec_title, df in sections.items()
        )

        return f"""
        {css}
        <div class="dakit-report">
            <div class="report-title">{html.escape(title)}</div>
            <div class="report-subtitle">{html.escape(subtitle)}</div>
            <div class="card-grid">{card_html}</div>
            {section_html}
        </div>
        """

    def _resolve_columns(self, columns: list[int | str] | None = None) -> list[str]:
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
            <a class="download-btn"
               download="{html.escape(file_name)}"
               href="data:text/html;base64,{encoded}">
                Download HTML Report
            </a>
        </div>
        """
        display(HTML(report_html + download_button))

    # ── Validators ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_header_names(header_names: list[str], dataframe: pd.DataFrame) -> None:
        if not isinstance(header_names, list):
            raise TypeError("header_names must be a list of strings.")
        if len(header_names) != len(dataframe.columns):
            raise ValueError("header_names must match the number of dataframe columns.")
        if not all(isinstance(n, str) for n in header_names):
            raise TypeError("All header_names must be strings.")

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")

    @staticmethod
    def _validate_align(value: str, name: str = "align") -> None:
        valid = ("left", "center", "right")
        if value not in valid:
            raise ValueError(f"{name} must be one of {valid}.")

    @staticmethod
    def _validate_margin(margin: dict[str, int]) -> None:
        if not isinstance(margin, dict):
            raise TypeError("margin must be a dictionary.")
        required = {"l", "r", "t", "b"}
        if not required.issubset(margin):
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
        elif isinstance(value, list):
            if not value:
                raise ValueError(f"{name} list cannot be empty.")
            for color in value:
                Table._validate_color(color, name)
        else:
            raise TypeError(f"{name} must be a string or a list of strings.")

    @staticmethod
    def _validate_font_family(value: str, name: str = "font_family") -> None:
        valid = ("Arial", "Calibri", "Helvetica", "Times New Roman", "Courier New", "Verdana")
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
        for w in column_widths:
            if not isinstance(w, int):
                raise TypeError("Each column width must be an integer.")
            if w <= 0:
                raise ValueError("Each column width must be greater than 0.")

    @staticmethod
    def _dtype_matches(dtype: pl.DataType, expected: str) -> bool:
        expected = str(expected).lower()
        if expected in ("number", "numeric", "int", "float"):
            return dtype.is_numeric()
        if expected in ("text", "string", "str"):
            return dtype in (pl.String, pl.Categorical, pl.Enum)
        if expected in ("bool", "boolean"):
            return dtype == pl.Boolean
        if expected in ("date", "datetime", "time"):
            return dtype in (pl.Date, pl.Datetime, pl.Time)
        return str(dtype).lower() == expected