#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
######################################
Convert Plate Layouts to Column Format
######################################

*Created on Tue, 05-May-2026 by A. Pahl*

Convert plate layouts from the row / column wide format to a columnar format."""

from pathlib import Path
import sys
import textwrap

import pandas as pd


def show_help():
    print(textwrap.dedent("""\
    This script converts plate layouts from the row / column wide format to a columnar format.  
    Optionally, the user can specify a custom column name for the values.  
    The result file will always be named "layout.tsv" and will be written to the same directory as the input file.

    Usage:
        $ ./layout_to_columns.py <input_file> [<column name>]
            
    Example:
        $ ./layout_to_columns.py plate.txt Cpd_Id
    """))


def process(fn: Path, col_name: str):
    base_dir = fn.parent
    rows = []
    columns = []
    wells = []
    values = []
    with open(fn, "r") as fo:
        for row_ctr, row in enumerate(fo, 1):
            row = row.strip("\r\n")
            if len(row) == 0:
                continue
            cols = row.split("\t")
            for col_ctr, col in enumerate(cols, 1):
                if len(col) == 0:
                    continue
                rows.append(row_ctr)
                columns.append(col_ctr)
                well = f"{chr(64 + row_ctr)}{col_ctr}"
                wells.append(well)
                values.append(col)
    df = pd.DataFrame({"Row": rows, "Column": columns, "Well": wells, col_name: values})
    df.to_csv(base_dir / "layout.tsv", sep="\t", index=False)
    print(f"Done. Output written to {base_dir / 'layout.tsv'}.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        show_help()
        sys.exit(1)
    if sys.argv[1] in ("-h", "--help"):
        show_help()
        sys.exit(0)
    fn = Path(sys.argv[1])
    if not fn.suffix == ".txt":
        print("Error: input file must be a .txt file.")
        sys.exit(1)
    col_name = "Value"
    if len(sys.argv) == 3:
        col_name = sys.argv[2]
    process(fn, col_name)
