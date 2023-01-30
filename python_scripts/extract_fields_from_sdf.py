#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############################################
Extract Fields from an SD File into a TSV File
##############################################

*Created on Mon, Jan 30, 2023 by A. Pahl*

Extract fields (listed as comma-separated) from a given SDF.
Run with `extract_fields.py <SD file> <comma-sep. list of fields>`."""

import sys
import os.path as op


def extract(fn, fields):
    base_fn = op.splitext(fn)[0]
    result_fn = base_fn + ".tsv"
    ctr = -0
    fields_in_order = []
    values = []
    first_entry = True
    add_value = False
    with open(fn) as f_in:
        with open(result_fn, "w") as f_out:
            for line in f_in:
                line = line.strip()

                if add_value:
                    values.append(line)
                    add_value = False
                    continue

                if line.startswith("$$$"):
                    assert len(values) == len(fields_in_order)
                    if first_entry:
                        f_out.write("\t".join(fields_in_order) + "\n")
                        first_entry = False
                    f_out.write("\t".join(values) + "\n")
                    ctr += 1
                    values = []
                    continue

                if line.startswith("> <"):
                    fld = line[3:-1]
                    if first_entry:
                        if fld in fields:
                            fields_in_order.append(fld)
                    if fld in fields_in_order:
                        add_value = True
                    continue

    print(f"Extracted {ctr} entries from {fn} into {result_fn}.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    fn = sys.argv[1]
    fields = sys.argv[2].split(",")
    extract(fn, fields)
