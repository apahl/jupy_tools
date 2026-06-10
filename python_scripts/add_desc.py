#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
########################
Add Descriptors to files
########################

*Created on Sun Dec 15 2024 17:45 by A. Pahl*

Add descriptors (and later maybe also fingerprints) to a set of files
containing Smiles and InChIKeys.
(The entries in each file are deduplicated by InChIKey.)
Changed on 20-Aug-2025: The structures are not deduplicated. Duplicate entries are logged, but not removed.
If the InChIKey is not present in the file, it is calculated from the Smiles,
but no standardization is performed on the structures.
Use the `stand_struct` script for this purpose before."""

import sys
import gzip
import csv
import base64 as b64
import argparse

import pandas as pd
import numpy as np

from rdkit.Chem import Mol
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem, QED
from rdkit.Chem.SpacialScore import SPS
from rdkit.Chem import Descriptors as Desc
from rdkit.Chem import rdMolDescriptors as rdMolDesc
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdReducedGraphs as ERG
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem import Fragments

# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsole

from Contrib.NP_Score import npscorer
from rdkit import RDLogger

LOG = RDLogger.logger()
LOG.setLevel(RDLogger.CRITICAL)
DEBUG = False


FSCORE = npscorer.readNPModel()


def score_np(mol):
    return npscorer.scoreMol(mol, FSCORE)


DESC = {
    "NP_Like": lambda x: round(score_np(x), 2),
    "QED": lambda x: round(QED.default(x), 3),
    "NumHA": lambda x: x.GetNumAtoms(),
    "MW": lambda x: round(Desc.MolWt(x), 2),
    "NumRings": rdMolDesc.CalcNumRings,
    "NumRingsArom": rdMolDesc.CalcNumAromaticRings,
    "NumRingsAli": rdMolDesc.CalcNumAliphaticRings,
    "NumHDon": rdMolDesc.CalcNumLipinskiHBD,
    "NumHAcc": rdMolDesc.CalcNumLipinskiHBA,
    "LogP": lambda x: round(Desc.MolLogP(x), 2),
    "TPSA": lambda x: round(rdMolDesc.CalcTPSA(x), 2),
    "NumRotBd": rdMolDesc.CalcNumRotatableBonds,
    "NumAtOx": lambda x: len([a for a in x.GetAtoms() if a.GetAtomicNum() == 8]),
    "NumAtN": lambda x: len([a for a in x.GetAtoms() if a.GetAtomicNum() == 7]),
    "NumAtHal": Fragments.fr_halogen,
    "NumAtBridgehead": rdMolDesc.CalcNumBridgeheadAtoms,
    "FCsp3": lambda x: round(rdMolDesc.CalcFractionCSP3(x), 3),
    "nSPS": lambda x: round(SPS(x), 2),  # normalizing is the default
}

NBITS = 2048
FPDICT = {}

EFP4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)
EFP6 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=NBITS)
FFP4 = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=NBITS,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)
FFP6 = rdFingerprintGenerator.GetMorganGenerator(
    radius=3,
    fpSize=NBITS,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)


def encode_arr(arr):
    s = " ".join(map(str, arr))
    raw = gzip.compress(s.encode("utf-8"))
    return b64.b64encode(raw).decode("ascii")


# Leaving this here for completion, but it is not used in this script. It can be used for decoding the fingerprints in the output file into numpy arrays again.
def decode_arr(text, dtype=np.int32):
    raw = b64.b64decode(text.encode("ascii"))
    s = gzip.decompress(raw).decode("utf-8")
    return np.fromstring(s, sep=" ", dtype=dtype)


FPDICT["ECFC4"] = lambda m: encode_arr(EFP4.GetCountFingerprintAsNumPy(m))
FPDICT["ECFC6"] = lambda m: encode_arr(EFP6.GetCountFingerprintAsNumPy(m))
FPDICT["ECFP4"] = lambda m: encode_arr(EFP4.GetFingerprintAsNumPy(m))
FPDICT["ECFP6"] = lambda m: encode_arr(EFP6.GetFingerprintAsNumPy(m))
FPDICT["FCFP4"] = lambda m: encode_arr(FFP4.GetFingerprintAsNumPy(m))
FPDICT["FCFP6"] = lambda m: encode_arr(FFP6.GetFingerprintAsNumPy(m))
FPDICT["ErG"] = lambda m: encode_arr(
    np.float32(ERG.GetErGFingerprint(m))
)  # Is directly a numpy array, no need for conversion
FPDICT["Pharm2D"] = lambda m: encode_arr(
    Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory).ToList()
)

# TODO: Add for v2:
#   Different sets of fingerprints (ECFC4, ECFC6, ECFP4, ECFP6, FCFP4, FCFP6, ErG, Pharmacophore)


def check_mol(mol: Mol) -> bool:
    """Check whether mol is indeed an instance of RDKit mol object,
    and not np.nan or None."""
    return isinstance(mol, Mol)


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
    for row in reader:
        for col in columns:
            # Clean up the strings:
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
            d[prop] = get_value(row[prop])
        yield d


def process(
    fn: str,
    desc_list: list[str],
    verbose: bool,
):

    desc_str = ""
    desc_main = "main" in desc_list
    if desc_main:
        desc_list.remove("main")
    if desc_main:
        desc_str += "desc"
    for dsc in desc_list:
        desc_str += f"_{dsc.lower()}"
    header = []
    hd_set = set()
    ctr_columns = ["In", "Out", "Fail_NoMol", "Duplicates", "Filter"]
    ctr = {x: 0 for x in ctr_columns}
    first_mol = True
    calc_inchi = False
    inchi_keys = set()
    fn = fn.split(",")  # allow comma separated list of files
    first_dot = fn[0].find(".")
    fn_base = fn[0][:first_dot]
    out_fn = f"{fn_base}_{desc_str}.tsv"
    outfile = open(out_fn, "w", encoding="utf-8")
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
        if ".csv" in f:
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
                file_obj = open(f, "r", encoding="utf-8")
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
                header = [x for x in rec if not x in {"Mol", "Smiles", "InChIKey"}]
                if desc_main:
                    header.extend(sorted(DESC.keys()))
                for dsc in desc_list:
                    header.append(dsc)
                # Put Smiles and InChIKey at the end:
                header.extend(["Smiles", "InChIKey"])
                if not "InChIKey" in rec:
                    calc_inchi = True
                    if DEBUG:
                        print("Calculating InChIKeys.")
                hd_set = set(header)
                outfile.write("\t".join(header) + "\n")

            mol_props = set()
            d = {}
            for prop in rec:
                if prop in hd_set:
                    if prop == "Mol":
                        continue
                    mol_props.add(prop)
                    d[prop] = rec[prop]

            # append "" to the missing props that were not in the mol:
            missing_props = hd_set - mol_props
            for prop in missing_props:
                d[prop] = ""

            if calc_inchi:
                try:
                    inchi = Chem.inchi.MolToInchiKey(mol)
                    d["InChIKey"] = inchi
                except:
                    ctr["Fail_NoMol"] += 1
                    continue
            else:
                inchi = d["InChIKey"]
            if inchi in inchi_keys:
                if DEBUG:
                    line = "\t".join([str(rec[x]) for x in rec if x != "Mol"])
                    print(f"\n\nDuplicate InChIKey: {inchi} for {line}")
                ctr["Duplicates"] += 1
            inchi_keys.add(inchi)

            # Finally calculate the descriptors:
            if desc_main:
                for desc in DESC:
                    # XXXX
                    try:
                        d[desc] = DESC[desc](mol)
                    except:
                        ctr["Fail_NoMol"] += 1
                        if DEBUG:
                            print(f"\nFailed to calculate {desc} for {d['Smiles']}")
                        continue
            # Also calculate the gzipped base64 encoded fingerprints:
            for dsc in desc_list:
                try:
                    d[dsc] = FPDICT[dsc](mol)
                except:
                    ctr["Fail_NoMol"] += 1
                    if DEBUG:
                        print(f"\nFailed to calculate {dsc} for {d['Smiles']}")
                    continue

            ctr["Out"] += 1
            line = [str(d[x]) for x in header]
            outfile.write("\t".join(line) + "\n")

            if ctr["In"] % 1000 == 0:
                print(
                    f"{fn_info} In: {ctr['In']:8d}  Out: {ctr['Out']: 8d}  Failed: {ctr['Fail_NoMol']:5d}  "
                    f"Dupl: {ctr['Duplicates']:6d}       ",
                    end=end_char,
                )
                sys.stdout.flush()

        if do_close:
            file_obj.close()
    outfile.close()
    print(
        f"{fn_info} In: {ctr['In']:8d}  Out: {ctr['Out']: 8d}  Failed: {ctr['Fail_NoMol']:5d}  "
        f"Dupl: {ctr['Duplicates']:6d}   done.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""

Add descriptors (and later maybe also fingerprints) zo a set of files containin Smiles and InChIKeys.
The entries in each file are deduplicated by InChIKey.
If the InChIKey is not present in the file, it is calculated from the Smiles, 
but no standardization is performed on the structures and the Smiles is not changed.
Use the `stand_struct` script for this purpose before.The output will be a tab-separated text file with SMILES.
If the Smiles can not be transformed into a mol, the entry is skipped.
Input files can be CSV, TSV with the structures in a `Smiles` column. The files may be gzipped.

Example:
Add the set of "main" descriptors (these are a set of, well, descriptive descriptors),
that are useful for Machine Learning and PCA visualizations of datasets:
    $ ./add_desc.py drugbank.tsv --desc main
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "in_file",
        help="The optionally gzipped input file (CSV, TSV). Can also be a comma-separated list of file names.",
    )
    parser.add_argument(
        "--desc_main",
        action="store_true",
        help="Add set of main descriptors (default action when no arguments are given; DEPRECATED: use --desc main instead).",
    )
    parser.add_argument(
        "--desc",
        choices=["main"] + sorted(FPDICT.keys()),
        nargs="+",
        default=["main"],
        help="Add set of descriptors and fingerprints to calculate, SEPARATED BY SPACES. The fingerprints are b64 encoded gzipped numpy arrays and can be decoded with the `decode_fp` function of the `utils` / `simple_utils` modules into numpy arrays again. Default: main",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Turn on verbose status output.",
    )
    args = parser.parse_args()
    # args.in_file = "TEST"
    if args.desc_main:
        args.desc.append("main")
    args.desc = sorted(set(args.desc))  # remove duplicates

    process(
        args.in_file,
        args.desc,
        args.v,
    )
