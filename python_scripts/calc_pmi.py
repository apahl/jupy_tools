#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################
Standardize Structure Files
###########################

*Created on Tue Aug 31 2021 08:45  by A. Pahl*
Standardize and filter SD files, e.g. the ChEMBL dataset.

Molecules are excluded from the MPI calculation when they have either
  - more than one undefined stereocenter, or
  - one or more defined stereocenters and at least one undefined stereocenter
    (creating diastereomers)
"""

import csv, gzip, sys
import time

import argparse
import signal
from contextlib import contextmanager

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors as rdMolDesc
from rdkit import RDLogger

from jupy_tools import pmi

LOG = RDLogger.logger()
LOG.setLevel(RDLogger.CRITICAL)


# Timeout code is taken from José's NPFC project:
# https://github.com/mpimp-comas/npfc/blob/master/npfc/utils.py
def raise_timeout(signum, frame):
    """Function to actually raise the TimeoutError when the time has come."""
    raise TimeoutError


@contextmanager
def timeout(time):
    # register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # schedule the signal to be sent after time
    signal.alarm(time)
    # run the code block within the with statement
    try:
        yield
    except TimeoutError:
        pass  # exit the with statement
    finally:
        # unregister the signal so it won't be triggered if the timeout is not reached
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


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


def smiles_to_mol(smiles: str) -> Chem.Mol:
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


def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> str:
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


def csv_supplier(fo, dialect):
    reader = csv.DictReader(fo, dialect=dialect)
    for row in reader:
        mol = smiles_to_mol(row["Smiles"])
        if mol is None:
            yield {"Mol": None}
            continue
        d = {}
        for prop in row:
            if prop == "Smiles":
                continue
            d[prop] = get_value(row[prop])
        d["Mol"] = mol
        yield d


def sdf_supplier(fo):
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
            d[prop] = get_value(mol.GetProp(prop))

        for prop in mol.GetPropNames():
            d[prop] = get_value(mol.GetProp(prop))
            mol.ClearProp(prop)
        d["Mol"] = mol
        yield d


def process(
    fn: str,
    id_col: str,
    tout: int,
    verbose: bool,
    every_n: int,
):
    columns = [id_col, "PMI1", "PMI2", "Duration", "Smiles"]
    header = []
    ctr = {x: 0 for x in ["In", "Out", "Fail_NoMol", "UndefStereo", "Timeout"]}
    first_mol = True
    sd_props = set()
    fn = fn.split(",")  # allow comma separated list of files
    first_dot = fn[0].find(".")
    fn_base = fn[0][:first_dot]
    out_fn = f"{fn_base}_pmi.tsv"
    outfile = open(out_fn, "w")
    # Initialize reader for the correct input type

    if verbose:
        # Add file name info and print newline after each info line.
        fn_info = f"({fn_base})"
        end_char = "\n"
    else:
        fn_info = ""
        end_char = "\r"

    for f in fn:
        do_close = True
        if "sd" in f:
            if f.endswith(".gz"):
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "rb")
            reader = sdf_supplier(file_obj)
        elif "csv" in f:
            if f.endswith(".gz"):
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "r")
            reader = csv_supplier(file_obj, dialect="excel")
        elif "tsv" in f:
            if f.endswith(".gz"):
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "r")
            reader = csv_supplier(file_obj, dialect="excel-tab")
        else:
            raise ValueError(f"Unknown input file format: {f}")

        for rec in reader:
            ctr["In"] += 1
            mol = rec["Mol"]
            if mol is None:
                ctr["Fail_NoMol"] += 1
                continue
            # Check for undefined stereochemistry.
            # Skip molecules that have either
            #   more than one undefined stereocenter or
            #   more than 0 defined stereocenters and at least one undefined stereocenter
            #     (creating diastereomers)
            num_st_all = rdMolDesc.CalcNumAtomStereoCenters(mol)
            num_st_undef = rdMolDesc.CalcNumUnspecifiedAtomStereoCenters(mol)
            num_st_def = num_st_all - num_st_undef
            if num_st_def > 0:
                if num_st_undef > 0:
                    ctr["UndefStereo"] += 1
                    continue
            else:
                if num_st_undef > 1:
                    ctr["UndefStereo"] += 1
                    continue
            if first_mol:
                first_mol = False
                header = columns.copy()
                sd_props = set(header.copy())
                outfile.write("\t".join(header) + "\n")

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

            # Measure how long the calculation takes
            start = time.time()
            timed_out = True
            with timeout(tout):
                try:
                    pmi1, pmi2 = pmi.calc_pmi(mol, n_conformers=15, avg=3)
                    timed_out = False
                except:
                    pass
            if timed_out:
                ctr["Timeout"] += 1
                continue
            end = time.time()
            elapsed = round(end - start, 1)
            d["PMI1"] = pmi1
            d["PMI2"] = pmi2
            d["Smiles"] = mol_to_smiles(mol)
            d["Duration"] = elapsed
            ctr["Out"] += 1
            line = [str(d[x]) for x in header]
            outfile.write("\t".join(line) + "\n")

            if ctr["In"] % every_n == 0:
                print(
                    f"{fn_info}  In: {ctr['In']:8d}  Out: {ctr['Out']: 8d}  Failed: {ctr['Fail_NoMol']:6d}  "
                    f"UndefStereo: {ctr['UndefStereo']:6d}  Timeout: {ctr['Timeout']:6d}       ",
                    end=end_char,
                )
                sys.stdout.flush()

        if do_close:
            file_obj.close()
    outfile.close()
    print(
        f"{fn_info}  In: {ctr['In']:8d}  Out: {ctr['Out']: 8d}  Failed: {ctr['Fail_NoMol']:6d}  "
        f"UndefStereo: {ctr['UndefStereo']:6d}  Timeout: {ctr['Timeout']:6d}   done."
    )
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Calculation of Principal Moment of Inertia (PMI).
Generation of multiple conformations and averaging by Median.

Molecules are excluded from the MPI calculation when they have either
  - more than one undefined stereocenter, or
  - one or more defined stereocenters and at least one undefined stereocenter
    (creating diastereomers)

Example:
    `$ ./calc_pmi.py enamine_full.tsv`
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "in_file",
        help="The optionally gzipped input file (CSV, TSV or SDF). Can also be a comma-separated list of file names.",
    )
    parser.add_argument(
        "id_col",
        help=("The name of the ID columnoutput type, " "e.g. `Compound_Id` "),
    )
    parser.add_argument(
        "-t",
        type=int,
        default=30,
        help="Timeout in seconds for each molecule (default: 30).",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Turn on verbose status output.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=200,
        help="Show info every `N` records (default: 200).",
    )
    args = parser.parse_args()
    print(args)
    process(
        args.in_file,
        args.id_col,
        args.t,
        args.v,
        args.n,
    )
