#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################
Standardize Structure Files
###########################

*Created on Tue Aug 31 2021 08:45  by A. Pahl*

Standardize and filter SD files, e.g. the ChEMBL dataset."""

import os
import sys
import gzip
import csv
from copy import deepcopy
import argparse
import signal

# from contextlib import contextmanager
import subprocess

import pandas as pd

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Mol
import rdkit.Chem.Descriptors as Desc
from rdkit.Chem.Scaffolds import MurckoScaffold

# from rdkit.Chem.MolStandardize.rdMolStandardize import (
#     TautomerEnumerator,
#     CleanupParameters,
# )
from rdkit.Chem.MolStandardize.rdMolStandardize import (
    CleanupInPlace,
    RemoveFragmentsInPlace,
    IsotopeParentInPlace,
    TautomerParentInPlace,
    ChargeParentInPlace,
    
    StereoParentInPlace
)


from rdkit import RDLogger

LOG = RDLogger.logger()
LOG.setLevel(RDLogger.CRITICAL)
DEBUG = False


def check_mol(mol: Mol) -> bool:
    """Check whether mol is indeed an instance of RDKit mol object,
    and not np.nan or None."""
    return isinstance(mol, Mol)    


# Code for legacy tautomer canonicalizer:
# Timeout code is taken from José's NPFC project:
# https://github.com/mpimp-comas/npfc/blob/master/npfc/utils.py
def raise_timeout(signum, frame):
    """Function to actually raise the TimeoutError when the time has come."""
    raise TimeoutError


# Code for legacy tautomer canonicalizer:
# @contextmanager
# def timeout(time):
#     # register a function to raise a TimeoutError on the signal.
#     signal.signal(signal.SIGALRM, raise_timeout)
#     # schedule the signal to be sent after time
#     signal.alarm(time)
#     # run the code block within the with statement
#     try:
#         yield
#     except TimeoutError:
#         pass  # exit the with statement
#     finally:
#         # unregister the signal so it won't be triggered if the timeout is not reached
#         signal.signal(signal.SIGALRM, signal.SIG_IGN)


class TimeOut(object):
    """Context manager to raise a TimeoutError if a block of code takes too long."""

    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        # register a function to raise a TimeoutError on the signal.
        signal.signal(signal.SIGALRM, raise_timeout)
        # schedule the signal to be sent after time
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
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


def get_atom_set(mol):
    result = set()
    for at in mol.GetAtoms():
        result.add(at.GetAtomicNum())
    return result


def has_isotope(mol: Mol) -> bool:
    for at in mol.GetAtoms():
        if at.GetIsotope() != 0:
            return True
    return False


def csv_supplier(fo, dialect):
    reader = csv.DictReader(fo, dialect=dialect)
    for row in reader:
        if len(row["Smiles"]) == 0:
            yield {"Mol": None}
            continue
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
        if mol is None or mol.GetNumAtoms() == 0:
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
    out_type: str,
    canon: str,
    idcol: str,
    columns: str,  # comma separated list of columns to keep
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    keep_dupl: bool,
    deglyco: bool,
    verbose: bool,
    every_n: int,
):
    canon = canon.lower()
    assert canon in {
        "none",
        "rdkit",
        "cxcalc",
    }, "Invalid canonicalization method, must be one of 'none', 'rdkit', 'cxcalc'"
    medchem_atoms = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # 5: Boron
    # Setting CleanUpParameters to retain stereochem information:
    # This only works with the new tautomer enumerator (option `--canon=rdkit`).
    # cup = CleanupParameters()
    # cup.tautomerReassignStereo = True
    # cup.tautomerRemoveBondStereo = False
    # cup.tautomerRemoveSp3Stereo = False
    # te = TautomerEnumerator(cup)

    deglyco_str = ""
    if deglyco:
        deglyco_str = "_deglyco"
    canon_str = ""
    if canon == "none":
        canon_str = "_nocanon"
    elif canon == "rdkit":
        canon_str = "_rdkit"
    elif canon == "cxcalc":
        if len(idcol) == 0:
            print(
                "ERROR: ID column must be specified for cxcalc canonicalization. Use option `--idcol`."
            )
            sys.exit(1)
        canon_str = "_cxcalc"
    dupl_str = ""
    if keep_dupl:
        dupl_str = "_dupl"
    min_ha_str = ""
    max_ha_str = ""
    if "medchem" in out_type:
        if min_heavy_atoms != 3:
            min_ha_str = f"_minha_{min_heavy_atoms}"
        if max_heavy_atoms != 50:
            max_ha_str = f"_maxha_{min_heavy_atoms}"

    if len(columns) > 0:
        columns = set(columns.split(","))
    else:
        columns = set()
    header = []
    if deglyco:
        ctr_columns = [
            "In",
            "Out",
            "Fail_NoMol",
            "Deglyco",
            "Fail_Deglyco",
            "Duplicates",
            "Filter",
        ]
    else:
        ctr_columns = ["In", "Out", "Fail_NoMol", "Duplicates", "Filter"]
    if canon == "legacy":
        ctr_columns.append("Timeout")
    ctr = {x: 0 for x in ctr_columns}
    first_mol = True
    sd_props = set()
    inchi_keys = set()
    fn = fn.split(",")  # allow comma separated list of files
    first_dot = fn[0].find(".")
    fn_base = fn[0][:first_dot]
    out_fn = f"{fn_base}_{out_type}{deglyco_str}{canon_str}{dupl_str}{min_ha_str}{max_ha_str}.tsv"
    outfile = open(out_fn, "w")
    if canon == "cxcalc":
        cx_calc_input_fn = f"{fn_base}_cxcalc_input.csv"
        cx_calc_result_fn = f"{fn_base}_cxcalc_result.tsv"
        cx_calc_inp = open(cx_calc_input_fn, "w")

    # Initialize reader for the correct input type

    if verbose:
        # Add file name info and print newline after each info line.
        fn_info = f"({fn_base}) "
        end_char = "\n"
    else:
        fn_info = ""
        end_char = "\r"

    for f in fn:
        do_close = True
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
                file_obj = open(f, "r")
            reader = csv_supplier(file_obj, dialect="excel")
        elif ".tsv" in f:
            file_type = "TSV"
            if f.endswith(".gz"):
                mode = " (gzipped)"
                file_obj = gzip.open(f, mode="rt")
            else:
                file_obj = open(f, "r")
            reader = csv_supplier(file_obj, dialect="excel-tab")
        else:
            raise ValueError(f"Unknown input file format: {f}")
        print(f"Detected file type: {file_type}{mode}.")

        for rec in reader:
            ctr["In"] += 1
            mol = rec["Mol"]
            if not check_mol(mol):
                ctr["Fail_NoMol"] += 1
                continue
            if first_mol:
                first_mol = False
                header = [x for x in rec if x != "Mol"]
                if len(columns) > 0:
                    header = [x for x in header if x in columns]
                header.append("InChIKey")
                sd_props = set(header.copy())
                header.append("Smiles")
                if canon == "cxcalc":
                    if idcol not in header:
                        print(
                            f"Id column `{idcol}` was not found in the dataset. It must be present for cxcalc canonicalization"
                        )
                        return
                    cx_calc_inp.write(f"{idcol},Smiles\n")
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

            # Standardization

            CleanupInPlace(mol)
            if not check_mol(mol):
                ctr["Fail_NoMol"] += 1
                continue
            RemoveFragmentsInPlace(mol)
            if not check_mol(mol):
                ctr["Fail_NoMol"] += 1
                continue
            try:
                # This can give an exception
                ChargeParentInPlace(mol)
            except:
                ctr["Fail_NoMol"] += 1
                continue
            if not check_mol(mol):
                ctr["Fail_NoMol"] += 1
                continue

            # Racemize all stereocenters
            if "rac" in out_type:
                StereoParentInPlace(mol)
                if not check_mol(mol):
                    ctr["Fail_NoMol"] += 1
                    continue

            if deglyco:
                # Deglycosylation should not lead to failed mols.
                # In case of error, the original mol is kept.
                mol_copy = deepcopy(mol)
                num_atoms = mol.GetNumAtoms()
                try:
                    mol = deglycosylate(mol)
                except ValueError:
                    ctr["Fail_Deglyco"] += 1
                    mol = mol_copy
                if not check_mol(mol):
                    ctr["Fail_Deglyco"] += 1
                    mol = mol_copy
                if mol.GetNumAtoms() < num_atoms:
                    ctr["Deglyco"] += 1

            if "murcko" in out_type:
                mol = MurckoScaffold.GetScaffoldForMol(mol)
                if not check_mol(mol):
                    ctr["Fail_NoMol"] += 1
                    continue

            if canon == "none" or canon == "cxcalc":
                # When canonicalization is not performed,  or when `cxcalc` is used,
                # we can check for duplicates already here.
                # When cxcalc is used, a final deduplication step has to be performed at the end.
                try:
                    inchi = Chem.inchi.MolToInchiKey(mol)
                except:
                    ctr["Fail_NoMol"] += 1
                    continue
                if not keep_dupl:
                    if inchi in inchi_keys:
                        ctr["Duplicates"] += 1
                        continue
                    inchi_keys.add(inchi)
                d["InChIKey"] = inchi

            # MedChem filters:
            if "medchem" in out_type:
                # Only MedChem atoms:
                if len(get_atom_set(mol) - medchem_atoms) > 0:
                    ctr["Filter"] += 1
                    continue
                # No isotopes:
                if has_isotope(mol):
                    ctr["Filter"] += 1
                    continue
                # HeavyAtom >= 3 or <= 50:
                ha = Desc.HeavyAtomCount(mol)
                if ha < min_heavy_atoms or ha > max_heavy_atoms:
                    ctr["Filter"] += 1
                    continue

            if canon == "rdkit":
                # Late canonicalization, because it is so expensive:
                TautomerParentInPlace(mol)        
                if not check_mol(mol):
                    ctr["Fail_NoMol"] += 1
                    continue
                try:
                    inchi = Chem.inchi.MolToInchiKey(mol)
                except:
                    ctr["Fail_NoMol"] += 1
                    continue
                if not keep_dupl:
                    # When canonicalization IS performed,
                    # we have to check for duplicates now:
                    if inchi in inchi_keys:
                        ctr["Duplicates"] += 1
                        continue
                    inchi_keys.add(inchi)
                d["InChIKey"] = inchi

            smi = mol_to_smiles(mol)
            if smi is None:
                ctr["Fail_NoMol"] += 1
                continue
            d["Smiles"] = smi
            ctr["Out"] += 1
            line = [str(d[x]) for x in header]
            outfile.write("\t".join(line) + "\n")
            if canon == "cxcalc":
                cx_calc_inp.write(f"{d[idcol]},{smi}\n")

            if ctr["In"] % every_n == 0:
                if canon == "legacy":
                    timeout_str = f"  Timeout: {ctr['Timeout']:6d}"
                else:
                    timeout_str = ""
                if deglyco:
                    print(
                        f"{fn_info} In: {ctr['In']:8d}  Out: {ctr['Out']: 8d}  Failed: {ctr['Fail_NoMol']:5d}  Deglyco: {ctr['Deglyco']:6d}  Fail_Deglyco: {ctr['Fail_Deglyco']:4d}  "
                        f"Dupl: {ctr['Duplicates']:6d}  Filt: {ctr['Filter']:6d}{timeout_str}       ",
                        end=end_char,
                    )
                else:
                    print(
                        f"{fn_info} In: {ctr['In']:8d}  Out: {ctr['Out']: 8d}  Failed: {ctr['Fail_NoMol']:5d}  "
                        f"Dupl: {ctr['Duplicates']:6d}  Filt: {ctr['Filter']:6d}{timeout_str}       ",
                        end=end_char,
                    )
                sys.stdout.flush()

        if do_close:
            file_obj.close()
    if canon == "cxcalc":
        cx_calc_inp.close()
    outfile.close()
    if canon == "legacy":
        timeout_str = f"  Timeout: {ctr['Timeout']:6d}"
    else:
        timeout_str = ""
    if deglyco:
        print(
            f"{fn_info} In: {ctr['In']:8d}  Out: {ctr['Out']: 8d}  Failed: {ctr['Fail_NoMol']:5d}  Deglyco: {ctr['Deglyco']:6d}  Fail_Deglyco: {ctr['Fail_Deglyco']:4d}  "
            f"Dupl: {ctr['Duplicates']:6d}  Filt: {ctr['Filter']:6d}{timeout_str}   done.",
        )
    else:
        print(
            f"{fn_info} In: {ctr['In']:8d}  Out: {ctr['Out']: 8d}  Failed: {ctr['Fail_NoMol']:5d}  "
            f"Dupl: {ctr['Duplicates']:6d}  Filt: {ctr['Filter']:6d}{timeout_str}   done.",
        )
    print("")
    if canon == "cxcalc":
        print(f"{fn_info}Calling cxcalc to generate tautomers...")
        try:
            subprocess.run(
                [
                    "cxcalc",
                    "-i",
                    idcol,
                    "-o",
                    cx_calc_result_fn,
                    cx_calc_input_fn,
                    "majortautomer",
                    "-H",
                    "7.4",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"{fn_info}Cxcalc failed with exit code {e.returncode}.")
            sys.exit(1)
        print(f"{fn_info}Merging data...")
        df_in = pd.read_csv(out_fn, sep="\t")
        df_in = df_in.rename(columns={"Smiles": "Smiles_orig"})
        df_cxcalc = pd.read_csv(cx_calc_result_fn, sep="\t")
        df_cxcalc = df_cxcalc.rename(columns={"structure": "Smiles_cxcalc"})
        df = pd.merge(df_in, df_cxcalc, on=idcol, how="left")
        assert len(df) == len(
            df_in
        ), f"{fn_info}Length of merged dataframe does not match."
        tmp = df[df["Smiles_cxcalc"].isna()]
        if len(tmp) > 0:
            print(
                f"{fn_info}{len(tmp)} records failed canonicalization in cxcalc. Original Smiles will be used for these."
            )
            if DEBUG:
                cx_calc_failed_fn = f"{fn_base}_cxcalc_failed.tsv"
                tmp.to_csv(cx_calc_failed_fn, sep="\t", index=False)
        df["Smiles"] = df["Smiles_cxcalc"].fillna(df["Smiles_orig"])
        df = df.drop(columns=["Smiles_orig", "Smiles_cxcalc", "InChIKey"], axis=1)
        print(f"{fn_info}Re-calculating InChIKeys...")
        prev_len = len(df)
        df = utils.calc_from_smiles(df, "CanSmiles", Chem.MolToSmiles)
        df = df.drop(columns=["Smiles"], axis=1)
        df = df.rename(columns={"CanSmiles": "Smiles"})
        df = inchi_from_smiles(df, filter_nans=True)
        if len(df) - prev_len > 0:
            print(
                f"{fn_info}{len(df) - prev_len} records were removed due to missing InChIKeys."
            )

        if not keep_dupl:
            print(f"{fn_info}Removing duplicates...")
            df = df.drop_duplicates(subset=["InChIKey"])

        print(f"{fn_info}Writing final data ({len(df)} records)...")
        df.to_csv(out_fn, sep="\t", index=False)
        if not DEBUG:
            print(f"{fn_info}Removing temporary files...")
            os.remove(cx_calc_input_fn)
            os.remove(cx_calc_result_fn)
        print(f"{fn_info}Merging done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Standardize structures. Input files can be CSV, TSV with the structures in a `Smiles` column
or an SD file. The files may be gzipped.
All entries with failed molecules will be removed.
By default, duplicate entries will be removed by InChIKey (can be turned off with the `--keep_dupl` option)
and structure canonicalization using the RDKit will be performed (can be turned with the `--canon=none` option).
Omitting structure canonicalization drastically reduces the runtime of the script.
Structures that fail the deglycosylation step WILL NOT BE REMOVED and the original structure is kept.
The output will be a tab-separated text file with SMILES.

Example:
Standardize the ChEMBL SDF download (gzipped), keep only MedChem atoms
and molecules between 3-50 heavy atoms, do not perform canonicalization:
    $ ./stand_struct.py chembl_29.sdf.gz medchemrac --canon=none
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "in_file",
        help="The optionally gzipped input file (CSV, TSV or SDF). Can also be a comma-separated list of file names.",
    )
    parser.add_argument(
        "output_type",
        choices=[
            "full",
            "fullrac",
            "medchem",
            "medchemrac",
            "fullmurcko",
            "medchemmurcko",
            "fullracmurcko",
            "medchemracmurcko",
        ],
        help=(
            "The output type. "
            "'full': Full dataset, only standardized; "
            "'fullrac': Like 'full', but with stereochemistry removed; "
            "'fullmurcko', 'fullracmurcko: Like 'full' or 'fullrac', but structures are reduced to their Murcko scaffolds; "
            "'medchem': Dataset with MedChem filters applied, bounds for the number of heavy atoms can be optionally given; "
            "'medchemrac': Like 'medchem', but with stereochemistry removed; "
            "'medchemmurcko', 'medchemracmurcko': Like 'medchem' or 'medchemrac', but structures are reduced to their Murcko scaffolds; "
            "(all filters, canonicalization and duplicate checks are applied after Murcko generation)."
        ),
    )
    parser.add_argument(
        "--canon",
        choices=["none", "rdkit", "cxcalc", "legacy"],
        default="rdkit",
        help="Select an algorithm for tautomer generation. `rdkit` uses the new C++ implementation from `rdMolStandardize.TautomerEnumerator`, `legacy` uses the older canonicalizer from `MolStandardize.tautomer`. `cxcalc` requires the ChemAxon cxcalc tool to be installed.",
    )
    parser.add_argument(
        "--idcol",
        type=str,
        default="",
        help="Name of the column that contains a unique identifier for the dataset. Required for canonicalization with `cxcalc`.",
    )
    parser.add_argument(
        "--nocanon",
        action="store_true",
        help="Do not perform canonicalization. DEPRECATED - use `--canon=none` instead.",
    )
    parser.add_argument(
        "--min_heavy_atoms",
        type=int,
        default=3,
        help="The minimum number of heavy atoms for a molecule to be kept (default: 3).",
    )
    parser.add_argument(
        "--max_heavy_atoms",
        type=int,
        default=50,
        help="The maximum number of heavy atoms for a molecule to be kept (default: 50).",
    )
    parser.add_argument(
        "-d", "--keep_duplicates", action="store_true", help="Keep duplicates."
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=str,
        default="",
        help="Comma-separated list of columns to keep (default: all).",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1000,
        help="Show info every `N` records (default: 1000).",
    )
    parser.add_argument(
        "--deglyco",
        action="store_true",
        help="deglycosylate structures. Requires jupy_tools.",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Turn on verbose status output.",
    )
    args = parser.parse_args()
    if args.nocanon:
        print(
            "NOTE: The `--nocanon` option is DEPRECATED - use `--canon=none` instead."
        )
        args.canon = "none"

    print(args)
    if args.deglyco:
        try:
            from jupy_tools.deglyco import deglycosylate
        except:
            print("deglycosylate() not found, please install jupy_tools.")
            sys.exit(1)
    if args.canon == "cxcalc":
        try:
            from jupy_tools import utils
            from jupy_tools.utils import inchi_from_smiles

            utils.TQDM = False
        except:
            print("inchi_from_smiles() not found, please install jupy_tools.")
            sys.exit(1)
    process(
        args.in_file,
        args.output_type,
        args.canon,
        args.idcol,
        args.columns,
        args.min_heavy_atoms,
        args.max_heavy_atoms,
        args.keep_duplicates,
        args.deglyco,
        args.v,
        args.n,
    )
