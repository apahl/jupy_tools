#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###################
Substructure Search
###################

*Created on Fri, 13-Feb-2026 by A. Pahl*

Perform a substructure search with one or multiple query molecules in arbitrary large files.
Multiple queries can be provided as comma-separated Smiles or Smarts strings."""

import sys
import gzip
import csv
from copy import deepcopy
import argparse


import pandas as pd

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Mol

from rdkit import RDLogger

WATERMARK = False
try:
    from watermark import watermark

    WATERMARK = True
except:
    pass
LOG = RDLogger.logger()
LOG.setLevel(RDLogger.CRITICAL)
DEBUG = False


def get_value(str_val):
    """convert a string into float or int, if possible."""
    if not str_val:
        return ""
    if str_val is None:
        return ""
    try:
        val = float(str_val)
        if "." not in str_val:
            val = int(val)
    except ValueError:
        val = str_val
    return val


def check_mol(mol: Mol) -> Mol | None:
    """Check whether mol is indeed an instance of RDKit mol object,
    and not np.nan or None.
    Make also sure that the mol can be round-tripped to Smiles and back."""
    if not isinstance(mol, Mol):
        return None
    smi = mol_to_smiles(mol)
    if smi is None:
        return None
    mol = smiles_to_mol(smi)
    return mol


def mol_to_smiles(mol: Mol, canonical: bool = True) -> str:
    """Generate Smiles from mol.

    Parameters:
    ===========
    mol: the input molecule
    canonical: whether to return the canonical Smiles or not

    Returns:
    ========
    The Smiles of the molecule (canonical by default). NAN for failed molecules."""

    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol, canonical=canonical)
        return smi
    except:
        return None


def smiles_to_mol(smiles: str) -> Mol:
    """Generate a RDKit Molecule from a Smiles.

    Parameters:
    ===========
    smiles: the input string

    Returns:
    ========
    The RDKit Molecule. If the Smiles parsing failed, NAN is returned instead.
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        return None
    except:
        return None


def csv_supplier(fo, dialect):
    reader = csv.DictReader(fo, dialect=dialect)
    columns = reader.fieldnames
    # Collect string properties, do not run them through `get_value()`:
    str_props = set()
    for row in reader:
        for col in columns:
            # Clean up Smiles strings:
            if col == "Smiles":
                if not isinstance(row[col], str):
                    row[col] = ""
                # Remove leading and trailing whitespace:
                row[col] = row[col].strip()
                # Remove newlines:
                row[col] = row[col].replace("\n", "")
                row[col] = row[col].replace("\r\n", "")
                # Also remove spaces:
                row[col] = row[col].replace(" ", "")
            # Clean up the remaining strings:
            row[col] = row[col].strip()
            row[col] = row[col].replace("\n", "; ")
            row[col] = row[col].replace("\r\n", "; ")
        d = {}
        if len(row["Smiles"]) == 0:
            d["Mol"] = None
        else:
            mol = smiles_to_mol(row["Smiles"])
            if mol is None:
                d["Mol"] = None
            else:
                d["Mol"] = mol
        for prop in row:
            if prop == "Smiles":
                continue
            val = row[prop]
            if not prop in str_props:
                val = get_value(val)
                if (
                    isinstance(val, str) and len(val) > 0
                ):  # missing value does not mean string prop
                    str_props.add(prop)
            d[prop] = val
        yield d
    print()
    print("Auto-detected string properties in the TSV:")
    print(", ".join(list(str_props)))


def sdf_supplier(fo):
    # Collect string properties, do not run them through `get_value()`:
    str_props = set()
    reader = Chem.ForwardSDMolSupplier(fo)
    for mol in reader:
        if mol is None:
            yield {"Mol": None}
            continue

        d = {}
        # Is the SD file name property used?
        name = mol.GetProp("_Name")
        if len(name) > 0:
            d["Name"] = get_value(name)
        for prop in mol.GetPropNames():
            try:
                val = mol.GetProp(prop)
            except UnicodeDecodeError:
                val = "<UnicodeDecodeError>"
            # Clean up the strings (1):
            val = val.strip()
            if not prop in str_props:
                val = get_value(val)
                if (
                    isinstance(val, str) and len(val) > 0
                ):  # missing value does not mean string prop
                    str_props.add(prop)
            # Clean up the strings (2):
            if prop in str_props:
                val = val.replace("\n", "; ")
                val = val.replace("\r\n", "; ")
                val = val.replace("\t", "; ")
            d[prop] = val
            mol.ClearProp(prop)
        if mol.GetNumAtoms() == 0:
            d["Mol"] = None
        else:
            d["Mol"] = mol
        yield d
    print()
    print("Auto-detected string properties in the SDF:")
    print(", ".join(list(str_props)))


def process(
    fn: str,
    query: str,
    format: str,
    add_h: str,
    keep_dupl: bool,
    every_n: int,
):
    def _sss(m, qm):
        if add_h_bool:
            m = Chem.AddHs(m)
        if m.HasSubstructMatch(qm):
            return True
        return False

    inchi_keys_seen = set()  # to track duplicates across queries
    query = query.strip()
    queries = query.split(",")
    num_queries = len(queries)
    formats = format.split(",")
    if len(formats) != 1 and len(formats) != num_queries:
        raise ValueError(
            f"Number of formats ({len(formats)}) has to be 1 or match the number of queries ({num_queries})."
        )
    if len(formats) == 1:
        formats = formats * num_queries

    add_h_bool = False
    if add_h == "auto":
        if "#1" in queries[0]:
            add_h_bool = True
    elif add_h == "yes":
        add_h_bool = True
    query_mols = []
    for query, format in zip(queries, formats):
        try:
            if format == "smarts":
                query_mol = Chem.MolFromSmarts(query)
            else:
                query_mol = Chem.MolFromSmiles(query)
        except:
            print(f"Failed to parse query Smiles/Smarts: {query}")
            return

        if query_mol is None:
            print(f"Failed to parse query Smiles/Smarts: {query}")
            return
        query_mols.append(query_mol)

    fn = fn.split(",")  # allow comma separated list of files
    for f_idx, f in enumerate(fn):
        first_rec = True
        first_hit = [True] * num_queries
        outfiles = [None] * num_queries
        sd_props = set()
        first_dot = f.find(".")
        fn_base = f[:first_dot]
        ctr_in = {f_idx: 0}
        ctr_hits = {x: 0 for x in range(num_queries)}
        ctr_dupl = 0
        ctr_failed = 0

        file_type = ""
        mode = ""
        if ".sd" in f:
            file_type = "SDF"
            if f.endswith(".gz"):
                mode = " (gzipped)"
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "rb")
            reader = sdf_supplier(file_obj)
        elif ".csv" in f:
            file_type = "CSV"
            if f.endswith(".gz"):
                mode = " (gzipped)"
                file_obj = gzip.open(f, mode="rt")
            else:
                file_obj = open(f, "r", encoding="utf-8")
            reader = csv_supplier(file_obj, dialect="excel")
        elif ".tsv" in f:
            file_type = "TSV"
            if f.endswith(".gz"):
                mode = " (gzipped)"
                file_obj = gzip.open(f, mode="rt")
            else:
                file_obj = open(f, "r", encoding="utf-8")
            reader = csv_supplier(file_obj, dialect="excel-tab")
        else:
            raise ValueError(f"Unknown input file format: {f}")
        print(f"Detected input file type: {file_type}{mode}.")

        for rec in reader:
            ctr_in[f_idx] += 1
            mol = rec["Mol"]
            if mol is None:
                ctr_failed += 1
                continue
            mol = check_mol(mol)
            if mol is None:
                ctr_failed += 1
                continue

            if ctr_in[f_idx] % every_n == 0:
                hits_info = " | ".join(
                    [f"Q{x}: {ctr_hits[x]:4d}" for x in range(num_queries)]
                )
                print(
                    f"In ({f_idx}): {ctr_in[f_idx]:8d} | {hits_info} | Dupl: {ctr_dupl:5d} | Failed: {ctr_failed:4d}  ",
                    end="\r",
                )
                sys.stdout.flush()

            if first_rec:
                first_rec = False
                header = [x for x in rec if x != "Mol"]
                sd_props = set(header.copy())
                header.append("Smiles")

            for q_idx, query_mol in enumerate(query_mols):
                if not _sss(mol, query_mol):
                    continue

                if not keep_dupl:
                    inchi_key = rec.get("InChIKey", None)
                    if inchi_key is not None:
                        if inchi_key in inchi_keys_seen:
                            ctr_dupl += 1
                            break  # No need to check the remaining queries for this record, since it is already a duplicate
                        inchi_keys_seen.add(inchi_key)

                if first_hit[q_idx]:
                    first_hit[q_idx] = False
                    out_fn = f"{fn_base}_sss_{q_idx}.tsv"
                    outfiles[q_idx] = open(out_fn, "w", encoding="utf-8")

                    outfiles[q_idx].write("\t".join(header) + "\n")

                mol_props = set()
                d = {}
                for prop in rec:
                    if prop in sd_props:
                        if prop == "Mol":
                            continue
                        mol_props.add(prop)
                        d[prop] = rec[prop]

                # append "" to the missing props that were not in the mol:
                missing_props = sd_props - mol_props
                for prop in missing_props:
                    d[prop] = ""

                smi = mol_to_smiles(mol)
                if smi is None:
                    ctr_failed += 1
                    continue
                d["Smiles"] = smi
                ctr_hits[q_idx] += 1
                line = [str(d[x]) for x in header]
                outfiles[q_idx].write("\t".join(line) + "\n")

        for outfile in outfiles:
            if outfile is not None:
                outfile.close()

        hits_info = " | ".join([f"Q{x}: {ctr_hits[x]:4d}" for x in range(num_queries)])
        print(
            f"In ({f_idx}): {ctr_in[f_idx]:8d} | {hits_info} | Dupl: {ctr_dupl:5d} | Failed: {ctr_failed:4d}  "
        )
        print(" Done.")
        sys.stdout.flush()


if __name__ == "__main__":
    if WATERMARK:
        print(
            "VERSIONS (by watermark):",
            "========================",
            watermark(packages="rdkit,pandas,jupy_tools", python=True),
            "========================",
            sep="\n",
        )
    parser = argparse.ArgumentParser(
        description="""
This script performs a substructure search in large files. The input files can be CSV, TSV with the structures in a `Smiles` column or an SD file. The files may be gzipped. Comma-separated lists of file names are also accepted. 
The query can be given as a single or multiple comma-separated Smiles (default) or Smarts strings. The script will read the input file(s) in a streaming fashion, so it can handle files that are larger than the available memory.
For multiple queries, separate output files will be generated for each query. The output will be a tab-separated text file with SMILES and all other columns from the input file (except the `Mol` column). The output file names are derived from the input file name by appending `_sss_<query_index>.tsv` to the base name.

Usage:
    $ ./sss.py <input_file> <query> [options]
          
Examples:
    # Single query:
    $ ./sss.py chembl_29.sdf.gz "c1ccccc1" -f smiles --addh auto -v -n 1000
    
    # Multiple queries:
    $ ./sss.py chembl_29.sdf.gz "c1ccccc1,c1ccncc1" -f smiles --addh auto -v -n 1000
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "in_file",
        help="The optionally gzipped input file (CSV, TSV or SDF). Can also be a comma-separated list of file names.",
    )
    parser.add_argument(
        "query",
        help="The query as a Smiles (default) or Smarts string. For Smarts, use the `-f smarts` flag. Multiple queries can be provided as comma-separated strings (e.g., 'c1ccccc1,c1ccncc1'). Each query will generate its own output file.",
    ),
    parser.add_argument(
        "-f",
        "--format",
        choices=["smiles", "smarts"],
        default="smiles",
        help="Whether the query is a Smiles (default) or Smarts string. For multiple queries, you can specify a single format (applies to all) or a comma-separated list of formats matching the number of queries (e.g., 'smiles,smarts').",
    )
    parser.add_argument(
        "--addh",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Wether to add explicit hydrogens to the searched molecules.",
    )
    parser.add_argument(
        "-d",
        "--duplicates",
        action="store_true",
        help="Whether to keep duplicate hits in the output (default: False). If False, only the first hit from any query will be written to the output file. If True, all hits will be written, including duplicates. Duplicates will only be tracked when the input file contains an InChIKey column or record.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=5000,
        help="Show info every `N` records (default: 1000).",
    )

    args = parser.parse_args()
    print(args)

    process(
        args.in_file,
        args.query,
        args.format,
        args.addh,
        args.duplicates,
        args.n,
    )
