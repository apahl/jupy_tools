#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for parallel substructure searching.
A large file with structures is searched for substructures from a small file,
which is kept in memory.
"""

import argparse
import time

import pandas as pd
import numpy as np

import rdkit.Chem.Descriptors as Desc

from jupy_tools import utils as u


def process(large_file, query_file, id_col):
    start_time = time.time()
    header = [id_col, "Hit", "Query", "Coverage"]
    first_dot = large_file.find(".")
    fn_base = large_file[:first_dot]
    df_qry = pd.read_csv(query_file, sep="\t")
    df_qry["Mol"] = df_qry["Smiles"].map(u.smiles_to_mol)
    df_qry = df_qry.dropna(subset=["Mol"])
    df_large = pd.read_csv(large_file, sep="\t")
    out_fn = f"{fn_base}_hits.tsv"
    outfile = open(out_fn, "w")
    outfile.write("\t".join(header) + "\n")
    num_hits = 0
    for ctr, (_, rec_mol) in enumerate(df_large.iterrows(), 1):
        result = {}
        mol = u.smiles_to_mol(rec_mol["Smiles"])
        if mol is np.nan:
            continue
        for _, rec_qry in df_qry.iterrows():
            query = rec_qry["Mol"]
            match = mol.GetSubstructMatch(query)
            if len(match) == 0:
                continue
            num_hits += 1
            ha_count = Desc.HeavyAtomCount(mol)
            coverage = len(match) / ha_count
            result[id_col] = rec_mol[id_col]
            result["Hit"] = "True"
            result["Query"] = rec_qry["Smiles"]
            result["Coverage"] = round(coverage, 3)
            break
        else:  # no substructure from the query file gave a hit
            result[id_col] = rec_mol[id_col]
            result["Hit"] = "False"
            result["Query"] = ""
            result["Coverage"] = ""

        if ctr % 5000 == 0:
            print(f"({fn_base})  In: {ctr:7d}   Hits: {num_hits:7d}  ")

        line = [str(result[x]) for x in header]
        outfile.write("\t".join(line) + "\n")
    outfile.close()
    duration = int(time.time() - start_time)
    print(
        f"({fn_base})  In: {ctr:7d}   Hits: {num_hits:7d}  done. ({time.strftime('%Hh %Mm %Ss', time.gmtime(duration))})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Perform a parallel substructure searching.
            A large file with structures is searched for substructures from a smaller query file,
            which is kept in memory. The matching from the query_file will stop after the first hit,
            so the ordering of the query_file (e.g. by size) *influences the results*.
            Outputs a tsv file with the results for each input record (id_col, Hit, Query, Coverage).
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "large_file",
        help="The large file that is searched.",
    )
    parser.add_argument(
        "query_file",
        help="The query file with the substructures to use for searching, has to fit in memory.",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        required=True,
        help="The column containing the identifier (required).",
    )
    args = parser.parse_args()
    print(args)
    process(
        args.large_file,
        args.query_file,
        args.id_col,
    )
