import polars as pl
import numpy as np
import os
import datetime as dt
import plotly.graph_objects as go
import pandas as pd
import warnings
from typing import Literal


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
        dataframe,
        title: str | None = None,
        header_names: list[str] | None = None,
    ) -> None:

        self.original_dataframe = dataframe
        self.dataframe = self.to_pandas(dataframe)

        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Nothing to display.")

        if header_names is not None:
            self._validate_header_names(header_names, self.dataframe)

        self.title = title
        self.header_names = header_names

        self.margin = dict(l=20, r=20, t=60, b=20)
        self.title_align = 0.5

        self.width = 2000
        self.height = 1000

        self.header_fillcolor = ["#1f2937", "#374151"]
        self.header_textcolor = "#ffffff"
        self.header_align: AlignValue = "center"

        self.cell_fillcolor = ["#f9fafb", "#ffffff"]
        self.cell_textcolor = "#111827"
        self.cell_align: AlignValue = "left"

        self.paper_bgcolor = "#f3f4f6"

        self.header_fontsize = 14
        self.cell_fontsize = 11

        self.header_height = 35
        self.cell_height = 30

        self.header_font_family: FontFamily = "Arial"
        self.cell_font_family: FontFamily = "Arial"

        self.header_bold = True
        self.cell_bold = False

        self.header_italic = False
        self.cell_italic = False

        self.header_linecolor = "#d1d5db"
        self.cell_linecolor = "#e5e7eb"

        self.column_widths = None

    def set_layout(
        self,
        title: str | None = None,
        title_align: AlignValue | None = None,
        width: int | None = None,
        height: int | None = None,
        header_height: int | None = None,
        cell_height: int | None = None,
        margin: dict | None = None,
        column_widths: list[int] | None = None,
    ) -> None:

        if title is not None:
            if not isinstance(title, str):
                raise TypeError("title must be a string.")
            self.title = title

        if title_align is not None:
            self._validate_align(title_align, "title_align")

            if title_align == "left":
                self.title_align = 0.05
            elif title_align == "center":
                self.title_align = 0.5
            elif title_align == "right":
                self.title_align = 0.95

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

    def set_global_style(
        self,
        paper_bgcolor: str | None = None,
    ) -> None:

        if paper_bgcolor is not None:
            self._validate_color(paper_bgcolor, "paper_bgcolor")
            self.paper_bgcolor = paper_bgcolor

    def set_header_style(
        self,
        fillcolor=None,
        textcolor=None,
        align: AlignValue | None = None,
        linecolor=None,
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
        fillcolor=None,
        textcolor=None,
        align: AlignValue | None = None,
        linecolor=None,
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

    def interactive(self, rows_per_page: int = 20):
        self._validate_positive_int(rows_per_page, "rows_per_page")

        try:
            from google.colab import data_table

            return data_table.DataTable(
                self.dataframe,
                include_index=False,
                num_rows_per_page=rows_per_page,
            )

        except ImportError:
            from IPython.display import display, HTML

            display(HTML(self.dataframe.to_html(index=False)))

    def show(
        self,
        filename: str = "table",
        max_rows: int | None = 1000,
        show_index: bool = False,
    ) -> None:

        self._validate_filename(filename)

        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Nothing to display.")

        if max_rows is not None:
            self._validate_positive_int(max_rows, "max_rows")

        if not isinstance(show_index, bool):
            raise TypeError("show_index must be True or False.")

        df = self.dataframe.copy()

        if show_index:
            df = df.reset_index()

        if max_rows is not None and len(df) > max_rows:
            warnings.warn(
                f"Only showing the first {max_rows} rows. "
                "Use interactive() for larger datasets."
            )
            df = df.head(max_rows)

        headers = self.header_names or list(df.columns)

        if self.header_names is not None and not show_index:
            self._validate_header_names(self.header_names, df)

        headers = self._format_values(
            headers,
            bold=self.header_bold,
            italic=self.header_italic,
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

        layout_kwargs = dict(
            width=self.width,
            height=self.height,
            margin=self.margin,
            paper_bgcolor=self.paper_bgcolor,
        )

        if self.title:
            layout_kwargs["title"] = dict(
                text=self.title,
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

    def _get_title_xanchor(self) -> str:
        if self.title_align <= 0.3:
            return "left"
        elif self.title_align >= 0.7:
            return "right"
        return "center"

    def _make_row_colors(self, row_count: int) -> list[str]:
        colors = self.cell_fillcolor

        if isinstance(colors, str):
            return [colors] * row_count

        repeated = colors * (row_count // len(colors) + 1)
        return repeated[:row_count]

    @staticmethod
    def _format_values(values, bold: bool = False, italic: bool = False):
        formatted = [str(value) for value in values]

        if bold:
            formatted = [f"<b>{value}</b>" for value in formatted]

        if italic:
            formatted = [f"<i>{value}</i>" for value in formatted]

        return formatted

    @staticmethod
    def to_pandas(dataframe) -> pd.DataFrame:
        if isinstance(dataframe, pd.DataFrame):
            return dataframe

        if dataframe.__class__.__module__.startswith("polars"):
            return dataframe.to_pandas()

        if dataframe.__class__.__module__.startswith("cudf"):
            return dataframe.to_pandas()

        raise TypeError(f"Unsupported dataframe type: {type(dataframe)}")

    @staticmethod
    def _validate_header_names(header_names, dataframe: pd.DataFrame) -> None:
        if not isinstance(header_names, list):
            raise TypeError("header_names must be a list of strings.")

        if len(header_names) != len(dataframe.columns):
            raise ValueError(
                "header_names must match the number of dataframe columns."
            )

        if not all(isinstance(name, str) for name in header_names):
            raise TypeError("All header_names must be strings.")

    @staticmethod
    def _validate_positive_int(value, name: str) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")

    @staticmethod
    def _validate_align(value, name: str = "align") -> None:
        valid = ["left", "center", "right"]

        if value not in valid:
            raise ValueError(f"{name} must be one of {valid}.")

    @staticmethod
    def _validate_margin(margin) -> None:
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
    def _validate_color(value, name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string color value.")

        if not value.strip():
            raise ValueError(f"{name} cannot be empty.")

    @staticmethod
    def _validate_color_or_list(value, name: str) -> None:
        if isinstance(value, str):
            Table._validate_color(value, name)
            return

        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError(f"{name} list cannot be empty.")

            for color in value:
                Table._validate_color(color, name)

            return

        raise TypeError(f"{name} must be a string or a list of strings.")

    @staticmethod
    def _validate_font_family(value, name: str = "font_family") -> None:
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
    def _validate_bool(value, name: str) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be True or False.")

    @staticmethod
    def _validate_filename(filename: str) -> None:
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")

        if not filename.strip():
            raise ValueError("filename cannot be empty.")

    @staticmethod
    def _validate_column_widths(column_widths) -> None:
        if not isinstance(column_widths, list):
            raise TypeError("column_widths must be a list of integers.")

        if len(column_widths) == 0:
            raise ValueError("column_widths cannot be empty.")

        for width in column_widths:
            if not isinstance(width, int):
                raise TypeError("Each column width must be an integer.")

            if width <= 0:
                raise ValueError("Each column width must be greater than 0.")