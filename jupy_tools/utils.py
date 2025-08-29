#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for data calculation.
"""

import gzip
import os.path as op
from pathlib import Path
import datetime
from glob import glob
import subprocess
import tempfile
import uuid
import signal
from contextlib import contextmanager

from typing import Any, Callable, List, Set, Tuple, Union

import pandas as pd
from pandas.core.frame import DataFrame

import numpy as np

from .simple_utils import *

try:
    from rdkit.Chem import AllChem as Chem, QED
    from rdkit.Chem import Mol
    from rdkit.Chem import DataStructs
    from rdkit.Chem import rdqueries
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem import rdReducedGraphs as ERG
    import rdkit.Chem.Descriptors as Desc
    from rdkit.Chem.SpacialScore import SPS
    from rdkit.Chem import rdMolDescriptors as rdMolDesc
    from rdkit.Chem import Fragments
    from rdkit.Chem.Scaffolds import MurckoScaffold

    from rdkit.Chem.MolStandardize.rdMolStandardize import (
        CleanupInPlace,
        RemoveFragmentsInPlace,
        IsotopeParentInPlace,
        TautomerParentInPlace,
        ChargeParentInPlace,
        StereoParentInPlace,
    )

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

    FPDICT["ECFC4"] = lambda m: EFP4.GetCountFingerprint(m)
    FPDICT["ECFC6"] = lambda m: EFP6.GetCountFingerprint(m)
    FPDICT["ECFP4"] = lambda m: EFP4.GetFingerprint(m)
    FPDICT["ECFP6"] = lambda m: EFP6.GetFingerprint(m)
    FPDICT["FCFP4"] = lambda m: FFP4.GetFingerprint(m)
    FPDICT["FCFP6"] = lambda m: FFP6.GetFingerprint(m)

    from Contrib.NP_Score import npscorer
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.info")
    # rdBase.DisableLog("rdApp.warn")
    RDKIT = True

except ImportError:
    RDKIT = False
    print("RDKit not installed.")


# def check_mol(mol: Mol) -> bool:
#     """Check whether mol is indeed an instance of RDKit mol object,
#     and not np.nan or None."""
#     return isinstance(mol, Mol)
def check_mol(mol: Mol) -> Mol:
    """Check whether mol is indeed an instance of RDKit mol object,
    and not np.nan or None.
    Make also sure that the mol can be round-tripped to Smiles and back.
    Returns the mol or np.nan."""
    if not isinstance(mol, Mol):
        return np.nan
    smi = mol_to_smiles(mol)
    if smi is np.nan:
        return np.nan
    mol = smiles_to_mol(smi)
    return mol


def fp_ecfc4_from_smiles(smi):
    mol = smiles_to_mol(smi)
    if mol is np.nan:
        return np.nan
    fp = Chem.GetMorganFingerprint(mol, 2)
    return fp


def add_fps(
    df: pd.DataFrame, smiles_col="Smiles", fp_col="FP", fp_type="ECFC4"
) -> pd.DataFrame:
    """Add a Fingerprint column to the DataFrame.
    Available types: ECFC4 (default), ECFC6, ECFP4, ECFP6, FCFP4, FCFP6.
    """
    assert RDKIT, "RDKit is not installed."
    assert (
        fp_type in FPDICT
    ), f"Unknown fingerprint type: {fp_type}. Available fingerprints are: {', '.join(FPDICT.keys())}"

    def _calc_fp(smi):
        mol = smiles_to_mol(smi)
        if mol is np.nan:
            return np.nan
        fp = FPDICT[fp_type](mol)
        return fp

    df = df.copy()

    if TQDM and len(df) > MIN_NUM_RECS_PROGRESS:
        df[fp_col] = df[smiles_col].progress_apply(lambda x: _calc_fp(x))
    else:
        df[fp_col] = df[smiles_col].apply(lambda x: _calc_fp(x))
    return df


def add_erg_fps(df: pd.DataFrame, smiles_col="Smiles", prefix="ErG") -> pd.DataFrame:
    """Add a ErG fingerprint columns to the DataFrame.
    Because the bits are inherently explainable, each of the 315 positions
    gets its own column. This function resets the index.

    Parameters:
    ===========
    df: pd.DataFrame
        The input DataFrame
    smiles_col: str
        The column containing the SMILES strings. Default: "Smiles"
    prefix: str
        The prefix for the new columns. Can be None or an empty string. Default: "ErG"

    Returns:
    ========
    A DataFrame with the added 315 ErG fingerprint columns.
    """
    assert RDKIT, "RDKit is not installed."
    if prefix is None:
        prefix = ""

    # Generate the 315 explanations for the bits:
    # Note: this will only work when this bug is fixed in the RDKit:
    #       https://github.com/rdkit/rdkit/issues/8201
    fp_len = 315
    properties = ["D", "A", "+", "-", "Hf", "Ar"]
    positions = []
    prop_len = len(properties)
    for idx1 in range(prop_len):
        for idx2 in range(idx1, prop_len):
            for dist in range(1, 16):
                positions.append(
                    f"{prefix}_{properties[idx1]}_{properties[idx2]}_{dist}"
                )
    assert len(positions) == fp_len, f"Expected 315 positions, got {len(positions)}"

    def _calc_fp(smi):
        mol = smiles_to_mol(smi)
        if mol is np.nan:
            return np.full(fp_len, np.nan)
        fp = ERG.GetErGFingerprint(mol)
        return fp

    df = df.copy()
    df = df.reset_index(drop=True)

    if TQDM and len(df) > MIN_NUM_RECS_PROGRESS:
        df["_ErG_FP"] = df[smiles_col].progress_apply(lambda x: _calc_fp(x))
    else:
        df["_ErG_FP"] = df[smiles_col].apply(lambda x: _calc_fp(x))

    # Split the 315 bits into separate columns:
    df_fp_cols = pd.DataFrame(df["_ErG_FP"].tolist(), columns=positions)
    assert len(df) == len(
        df_fp_cols
    ), f"Length mismatch: {len(df)} != {len(df_fp_cols)}"

    # Remove the original column:
    df = df.drop(columns=["_ErG_FP"])

    # Merge with original DataFrame:
    df = pd.concat([df, df_fp_cols], axis=1)

    return df


def read_sdf(
    fn, keep_mols=True, merge_prop: str = None, merge_list: Union[List, Set] = None
) -> pd.DataFrame:
    """Create a DataFrame instance from an SD file.
    The input can be a single SD file or a list of files and they can be gzipped (fn ends with `.gz`).
    If a list of files is used, all files need to have the same fields.
    The molecules will be converted to Smiles and can optionally be stored as a `Mol` column.
    Records with no valid molecule will be dropped.

    Parameters:
    ===========
    merge_prop: A property in the SD file on which the file should be merge
        during reading.
    merge_list: A list or set of values on which to merge.
        Only the values of the list are kept.

    Returns:
    ========
    A Pandas DataFrame containing the structures as Smiles.
    """

    d = {"Smiles": []}
    if keep_mols:
        d["Mol"] = []
    ctr = {x: 0 for x in ["In", "Out", "Fail_NoMol"]}
    if merge_prop is not None:
        ctr["NotMerged"] = 0
    first_mol = True
    sd_props = set()
    if not isinstance(fn, list):
        fn = [fn]
    for f in fn:
        do_close = True
        if isinstance(f, str):
            if f.endswith(".gz"):
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "rb")
        else:
            file_obj = f
            do_close = False
        reader = Chem.ForwardSDMolSupplier(file_obj)
        for mol in reader:
            ctr["In"] += 1
            if not mol:
                ctr["Fail_NoMol"] += 1
                continue
            if first_mol:
                first_mol = False
                # Is the SD file name property used?
                name = mol.GetProp("_Name")
                if len(name) > 0:
                    has_name = True
                    d["Name"] = []
                else:
                    has_name = False
                for prop in mol.GetPropNames():
                    sd_props.add(prop)
                    d[prop] = []
            if merge_prop is not None:
                # Only keep the record when the `merge_prop` value is in `merge_list`:
                if get_value(mol.GetProp(merge_prop)) not in merge_list:
                    ctr["NotMerged"] += 1
                    continue
            mol_props = set()
            ctr["Out"] += 1
            for prop in mol.GetPropNames():
                if prop in sd_props:
                    mol_props.add(prop)
                    d[prop].append(get_value(mol.GetProp(prop)))
                mol.ClearProp(prop)
            if has_name:
                d["Name"].append(get_value(mol.GetProp("_Name")))
                mol.ClearProp("_Name")

            # append NAN to the missing props that were not in the mol:
            missing_props = sd_props - mol_props
            for prop in missing_props:
                d[prop].append(np.nan)
            d["Smiles"].append(mol_to_smiles(mol))
            if keep_mols:
                d["Mol"].append(mol)
        if do_close:
            file_obj.close()
    # Make sure, that all columns have the same length.
    # Although, Pandas would also complain, if this was not the case.
    d_keys = list(d.keys())
    if len(d_keys) > 1:
        k_len = len(d[d_keys[0]])
        for k in d_keys[1:]:
            assert k_len == len(d[k]), f"{k_len=} != {len(d[k])}"
    result = pd.DataFrame(d)
    print(ctr)
    if INTERACTIVE:
        info(result, "read_sdf")
    return result


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
        return np.nan
    try:
        smi = Chem.MolToSmiles(mol, canonical=canonical)
        return smi
    except:
        return np.nan


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
        if mol is not None and isinstance(mol, Mol):
            return mol
        return np.nan
    except:
        return np.nan


def add_mol_col(df: pd.DataFrame, smiles_col="Smiles") -> pd.DataFrame:
    """Add a column containing the RDKit Molecule.

    Parameters:
    ===========
    df: the input DataFrame
    smiles_col: the name of the column containing the Smiles

    Returns:
    ========
    A DataFrame with a column containing the RDKit Molecule.
    """

    if TQDM and len(df) > MIN_NUM_RECS_PROGRESS:
        df["Mol"] = df[smiles_col].progress_apply(smiles_to_mol)
    else:
        df["Mol"] = df[smiles_col].apply(smiles_to_mol)
    return df


def standardize_mol(
    mol,
    largest_fragment=True,
    uncharge=True,
    standardize=True,
    remove_stereo=False,
    canonicalize_tautomer=False,
) -> str:
    """Standardize the molecule structures.
    Returns:
    ========
    Smiles of the standardized molecule. NAN for failed molecules."""

    mol = check_mol(mol)
    if mol is np.nan:
        return np.nan
    CleanupInPlace(mol)
    mol = check_mol(mol)
    if mol is np.nan:
        return np.nan
    if largest_fragment:
        RemoveFragmentsInPlace(mol)
        mol = check_mol(mol)
        if mol is np.nan:
            return np.nan
    if uncharge:
        ChargeParentInPlace(mol)
        mol = check_mol(mol)
        if mol is np.nan:
            return np.nan
    if remove_stereo:
        StereoParentInPlace(mol)
        mol = check_mol(mol)
        if mol is np.nan:
            return np.nan
    if canonicalize_tautomer:
        try:
            TautomerParentInPlace(mol)
            mol = check_mol(mol)
        except:
            # This is debatable, but for now, when canonicalization fails, fail the molecule
            mol = np.nan
        if mol is np.nan:
            return np.nan
    return mol_to_smiles(mol)


def standardize_smiles(
    smiles,
    largest_fragment=True,
    uncharge=True,
    standardize=True,
    remove_stereo=False,
    canonicalize_tautomer=True,
) -> str:
    """Creates a molecule from the Smiles string and passes it to `standardize_mol().

    Returns:
    ========
    The Smiles string of the standardized molecule."""
    mol = smiles_to_mol(smiles)  # None handling is done in `standardize_mol`
    result = standardize_mol(
        mol,
        largest_fragment=largest_fragment,
        uncharge=uncharge,
        standardize=standardize,
        remove_stereo=remove_stereo,
        canonicalize_tautomer=canonicalize_tautomer,
    )
    return result


def standardize_df(df, smiles_col="Smiles", **kwargs) -> DataFrame:
    """Standardize the structures in the DataFrame.
    The Smiles column is replaced by the standardized Smiles.

    Parameters:
    ===========
    df: the input DataFrame
    smiles_col: the name of the column containing the Smiles

    Keyword arguments:
    ==================
    largest_fragment: bool
        Whether to keep only the largest fragment. Default: True
    uncharge: bool
        Whether to remove charges. Default: True
    standardize: bool
        Whether to standardize the molecule. Default: True
    remove_stereo: bool
        Whether to remove stereochemistry. Default: False

    Returns:
    ========
    A DataFrame with the standardized Smiles.
    """
    df = df.copy()
    if TQDM and len(df) > MIN_NUM_RECS_PROGRESS:
        df[smiles_col] = df[smiles_col].progress_apply(
            lambda x: standardize_smiles(x, **kwargs)
        )
    else:
        df[smiles_col] = df[smiles_col].apply(lambda x: standardize_smiles(x, **kwargs))
    return df


def add_desc(df, smiles_col="Smiles", filter_nans=True) -> DataFrame:
    """Add a set of standard RDKit descriptors to the DataFrame.
    The descriptors are added as new columns.
    if filter_nans is True, rows with NANs are removed.
    Returns: the DataFrame with the added columns."""

    fscore = npscorer.readNPModel()

    def _score_np(mol):
        return npscorer.scoreMol(mol, fscore)

    descriptors = {
        "NP_Like": lambda x: round(_score_np(x), 2),
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
    desc_keys = list(descriptors.keys())
    df = df.copy()
    for key in desc_keys:
        df = calc_from_smiles(
            df, key, descriptors[key], smiles_col, filter_nans=filter_nans
        )
    return df


def get_atom_set(mol):
    result = set()
    for at in mol.GetAtoms():
        result.add(at.GetAtomicNum())
    return result


def filter_mols(
    df: pd.DataFrame, filter: Union[str, List[str]], smiles_col="Smiles", **kwargs
) -> pd.DataFrame:
    """Apply different filters to the molecules.
    If the dataframe contains a `Mol` column, it will be used,
    otherwise the `Mol` column is generated from the `smiles_col` column.
    This might be problematic for large dataframes.
    When in doubt, use `filter_smiles` instead.

    Parameters:
    ===========
    filter [str or list of strings]: The name of the filter to apply.
        Available filters:
            - Isotopes: Keep only non-isotope molecules
            - MedChemAtoms: Keep only molecules with MedChem atoms
            - MinHeavyAtoms: Keep only molecules with 3 or more heavy atoms
            - MaxHeavyAtoms: Keep only molecules with 50 or less heavy atoms
            - Duplicates: Remove duplicates by InChiKey

    kwargs:
        provides the possibility to override the heavy atoms cutoffs:
            - min_heavy_atoms: int
            - max_heavy_atoms: int
    """
    available_filters = {
        "Isotopes",
        "MedChemAtoms",
        "MinHeavyAtoms",
        "MaxHeavyAtoms",
        "Duplicates",
    }
    medchem_atoms = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # 5: Boron
    min_heavy_atoms = kwargs.get("min_heavy_atoms", 3)
    max_heavy_atoms = kwargs.get("max_heavy_atoms", 50)

    def has_non_medchem_atoms(mol):
        if len(get_atom_set(mol) - medchem_atoms) > 0:
            return True
        return False

    def has_isotope(mol) -> bool:
        for at in mol.GetAtoms():
            if at.GetIsotope() != 0:
                return True
        return False

    shape1 = df.shape
    df = df.copy()
    if isinstance(filter, str):
        filter = [filter]
    for filt in filter:
        if filt not in available_filters:
            raise ValueError(f"Unknown filter: {filt}")
    calc_ha = False
    cols_to_remove = []
    if "Mol" not in df.keys():
        print("Adding molecules...")
        df["Mol"] = df[smiles_col].apply(smiles_to_mol)
        cols_to_remove.append("Mol")
    print(f"Applying filters ({len(filter)})...")
    df = df[~df["Mol"].isnull()]
    for filt in tqdm(filter):
        if filt == "Isotopes":
            # df = apply_to_smiles(df, smiles_col, {"FiltIsotopes": has_isotope})
            df["FiltIsotopes"] = df["Mol"].apply(has_isotope)
            df = df.query("FiltIsotopes == False")
            cols_to_remove.append("FiltIsotopes")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MedChemAtoms":
            # df = apply_to_smiles(
            #     df, smiles_col, {"FiltNonMCAtoms": has_non_medchem_atoms}
            # )
            df["FiltNonMCAtoms"] = df["Mol"].apply(has_non_medchem_atoms)
            df = df.query("FiltNonMCAtoms == False")
            cols_to_remove.append("FiltNonMCAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MinHeavyAtoms":
            if not calc_ha:
                # df = apply_to_smiles(
                #     df, smiles_col, {"FiltHeavyAtoms": Desc.HeavyAtomCount}
                # )
                df["FiltHeavyAtoms"] = df["Mol"].apply(Desc.HeavyAtomCount)
                calc_ha = True
            df = df.query(f"FiltHeavyAtoms >= {min_heavy_atoms}")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt} (cutoff: {min_heavy_atoms}): ", end="")
        elif filt == "MaxHeavyAtoms":
            if not calc_ha:
                # df = apply_to_smiles(
                #     df, smiles_col, {"FiltHeavyAtoms": Desc.HeavyAtomCount}
                # )
                df["FiltHeavyAtoms"] = df["Mol"].apply(Desc.HeavyAtomCount)
                calc_ha = True
            df = df.query(f"FiltHeavyAtoms <= {max_heavy_atoms}")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt} (cutoff: {max_heavy_atoms}): ", end="")
        elif filt == "Duplicates":
            # df = apply_to_smiles(
            #     df, smiles_col, {"FiltInChiKey": Chem.inchi.MolToInchiKey}
            # )
            df["FiltInChiKey"] = df["Mol"].apply(Chem.inchi.MolToInchiKey)
            df = df[df["FiltInChIKey"].notna()]
            df = df.drop_duplicates(subset="FiltInChiKey")
            cols_to_remove.append("FiltInChiKey")
            print(f"Applied filter {filt}: ", end="")
        else:
            print()
            raise ValueError(f"Unknown filter: {filt}.")
        print(len(df))
    df = drop_cols(df, cols_to_remove)
    shape2 = df.shape
    if INTERACTIVE:
        info(df, "filter_mols", f"{shape1[0] - shape2[0]:4d} rows removed.")
    return df


def filter_smiles(
    df: pd.DataFrame, filter: Union[str, List[str]], smiles_col="Smiles", **kwargs
) -> pd.DataFrame:
    """Apply different filters to the molecules.
    The molecules are generated from the `smiles_col` column on the fly and are not stored in the DF.
    Make sure, that the DF contains only valid Smiles, first.

    Parameters:
    ===========
    filter [str or list of strings]: The name of the filter to apply.
        Available filters:
            - Isotopes: Keep only non-isotope molecules
            - MedChemAtoms: Keep only molecules with MedChem atoms
            - MinHeavyAtoms: Keep only molecules with 3 or more heavy atoms
            - MaxHeavyAtoms: Keep only molecules with 50 or less heavy atoms
            - Duplicates: Remove duplicates by InChiKey

    kwargs:
        provides the possibility to override the heavy atoms cutoffs:
            - min_heavy_atoms: int
            - max_heavy_atoms: int
    """
    available_filters = {
        "Isotopes",
        "MedChemAtoms",
        "MinHeavyAtoms",
        "MaxHeavyAtoms",
        "Duplicates",
    }
    medchem_atoms = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # 5: Boron
    min_heavy_atoms = kwargs.get("min_heavy_atoms", 3)
    max_heavy_atoms = kwargs.get("max_heavy_atoms", 50)
    global INTERACTIVE
    interact_flag = INTERACTIVE
    INTERACTIVE = False  # Disble INFO messages

    def has_non_medchem_atoms(smiles) -> bool:
        mol = smiles_to_mol(smiles)
        if mol is np.nan:
            return True
        if len(get_atom_set(mol) - medchem_atoms) > 0:
            return True
        return False

    def has_isotope(smiles) -> bool:
        mol = smiles_to_mol(smiles)
        if mol is np.nan:
            return True
        for at in mol.GetAtoms():
            if at.GetIsotope() != 0:
                return True
        return False

    def ha(smiles) -> int:
        mol = smiles_to_mol(smiles)
        if mol is np.nan:
            return 0
        return Desc.HeavyAtomCount(mol)

    shape1 = df.shape
    df = df.copy()
    if isinstance(filter, str):
        filter = [filter]
    for filt in filter:
        if filt not in available_filters:
            raise ValueError(f"Unknown filter: {filt}")
    calc_ha = False
    cols_to_remove = []
    print(f"Applying filters ({len(filter)})...")
    df = df[~df["Smiles"].isnull()]
    for filt in tqdm(filter):
        if filt == "Isotopes":
            # df = apply_to_smiles(df, smiles_col, {"FiltIsotopes": has_isotope})
            df["FiltIsotopes"] = df[smiles_col].apply(has_isotope)
            df = df.query("FiltIsotopes == False")
            cols_to_remove.append("FiltIsotopes")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MedChemAtoms":
            # df = apply_to_smiles(
            #     df, smiles_col, {"FiltNonMCAtoms": has_non_medchem_atoms}
            # )
            df["FiltNonMCAtoms"] = df[smiles_col].apply(has_non_medchem_atoms)
            df = df.query("FiltNonMCAtoms == False")
            cols_to_remove.append("FiltNonMCAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MinHeavyAtoms":
            if not calc_ha:
                # df = apply_to_smiles(
                #     df, smiles_col, {"FiltHeavyAtoms": Desc.HeavyAtomCount}
                # )
                df["FiltHeavyAtoms"] = df[smiles_col].apply(ha)
                calc_ha = True
            df = df.query(f"FiltHeavyAtoms >= {min_heavy_atoms}")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt} (cutoff: {min_heavy_atoms}): ", end="")
        elif filt == "MaxHeavyAtoms":
            if not calc_ha:
                df["FiltHeavyAtoms"] = df[smiles_col].apply(ha)
                calc_ha = True
            df = df.query(f"FiltHeavyAtoms <= {max_heavy_atoms}")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt} (cutoff: {max_heavy_atoms}): ", end="")
        elif filt == "Duplicates":
            # df = apply_to_smiles(
            #     df, smiles_col, {"FiltInChiKey": Chem.inchi.MolToInchiKey}
            # )
            df = inchi_from_smiles(
                df, smiles_col=smiles_col, inchi_col="FiltInChiKey", filter_nans=True
            )
            df = df.drop_duplicates(subset="FiltInChiKey")
            cols_to_remove.append("FiltInChiKey")
            print(f"Applied filter {filt}: ", end="")
        else:
            print()
            raise ValueError(f"Unknown filter: {filt}.")
        print(len(df))
    df = drop_cols(df, cols_to_remove)

    INTERACTIVE = interact_flag  # Restore option for showing INFO messages
    if INTERACTIVE:
        shape2 = df.shape
        info(df, "filter_smiles", f"{shape1[0] - shape2[0]:4d} rows removed.")
    return df


def fr_stereo(mol, spec_only=False) -> float:
    """Calculate the fraction of stereogenic carbons.
    This is defined as number of carbons with defined or undefined stereochemistry,
    divided by the total number of carbons.
    With `spec_only=True`, only the number of carbons with defined stereochemistry
    is considered.

    Returns:
    ========
    The fraction of stereogenic carbons [0..1], rounded to 3 digits after the decimal.
    """
    num_spec = 0
    num_unspec = 0
    q = rdqueries.AtomNumEqualsQueryAtom(6)
    num_carbons = len(mol.GetAtomsMatchingQuery(q))
    if num_carbons == 0:
        return 0.0
    atoms = mol.GetAtoms()
    chiraltags = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True)
    # example output [(1, '?'), (2, 'R'), (5, 'S'), (8, 'S')]
    for tag in chiraltags:
        # Only consider carbon chirality:
        if atoms[tag[0]].GetAtomicNum() != 6:
            continue
        if tag[1] == "R" or tag[1] == "S":
            num_spec += 1
        else:
            if not spec_only:
                num_unspec += 1
    return round((num_spec + num_unspec) / num_carbons, 3)


def calc_from_smiles(
    df: pd.DataFrame, new_col, func: Callable, smiles_col="Smiles", filter_nans=True
) -> pd.DataFrame:
    """Calculate a new column that would require a molecule as input
    directly from a Smiles column.

    Parameters:
    ===========
    df: pd.DataFrame
        The dataframe to apply the function to.
    new_col: str
        The name of the new column to create.
    func: Callable
        The function to apply to the molecule column.
    smiles_col: str
        The name of the column containing the Smiles strings for the molecules.
    filter_nans: bool
        If True, remove rows with NaN values.
    """

    def _smiles_func(smiles):
        mol = smiles_to_mol(smiles)
        if mol is np.nan:
            return np.nan
        try:
            result = func(mol)
            return result
        except:
            return np.nan

    shape1 = df.shape
    df = df.copy()
    if TQDM and len(df) > MIN_NUM_RECS_PROGRESS:
        df[new_col] = df[smiles_col].progress_apply(_smiles_func)
    else:
        df[new_col] = df[smiles_col].apply(_smiles_func)
    if filter_nans:
        df = df[df[new_col].notna()]
        if INTERACTIVE:
            shape2 = df.shape
            info(
                df,
                "calc_from_smiles",
                f"{shape1[0] - shape2[0]:4d} rows removed because of nans.",
            )
    return df


def inchi_from_smiles(
    df: pd.DataFrame, smiles_col="Smiles", inchi_col="InChIKey", filter_nans=True
) -> pd.DataFrame:
    """Generate InChIKeys from Smiles.

    Parameters:
    ===========
    df: pd.DataFrame
        The dataframe to apply the function to.
    smiles_col: str
        The name of the column containing the Smiles strings.
    inchi_col: str
        The name of the column to store the InChIKeys (default: "InChIKey").
    filter_nans: bool
        If True, remove rows with NaN in the InChIKey column.

    Returns:
    ========
    pd.DataFrame
        The dataframe with the Murcko scaffolds and the InChIKeys of the scaffolds.
    """

    shape1 = df.shape
    df = df.copy()
    df = calc_from_smiles(
        df,
        inchi_col,
        Chem.inchi.MolToInchiKey,
        smiles_col=smiles_col,
        filter_nans=False,
    )
    if filter_nans:
        df = df[df[inchi_col].notna()]
        if INTERACTIVE:
            shape2 = df.shape
            info(df, "inchi_from_smiles", f"{shape1[0] - shape2[0]:4d} rows removed.")
    return df


def murcko_from_smiles(
    df: pd.DataFrame, smiles_col="Smiles", murcko_col="Murcko_Smiles", filter_nans=True
) -> pd.DataFrame:
    """Generate Murcko scaffolds from Smiles.
    Molecules without any rings do not have a Murcko scaffold,
    their Murcko_Smiles column will set to NaN.
    In addition to the Murcko_Smiles column, the Murcko_InChIKey column
    is also generated.

    Parameters:
    ===========
    df: pd.DataFrame
        The dataframe to apply the function to.
    smiles_col: str
        The name of the column containing the Smiles strings.
    murcko_col: str
        The name of the column to store the Murcko scaffolds as Smiles (default: "MurckoSmiles").
    filter_nans: bool
        If True, remove rows with NaN in the Murcko column (default: True).

    Returns:
    ========
    pd.DataFrame
        The dataframe with the Murcko scaffolds and the InChIKeys of the scaffolds.
    """

    def _murcko(mol):
        result = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if result == "" or result is None:
            return np.nan
        else:
            return result

    global INTERACTIVE
    interact_flag = INTERACTIVE
    shape1 = df.shape
    df = df.copy()
    # Temporariliy turn off INTERACTIVE output to avoid printing the info multiple times
    INTERACTIVE = False
    df = calc_from_smiles(
        df, murcko_col, _murcko, smiles_col=smiles_col, filter_nans=False
    )
    df = inchi_from_smiles(
        df, smiles_col=murcko_col, inchi_col="Murcko_InChIKey", filter_nans=False
    )
    INTERACTIVE = interact_flag  # restore previous value
    if filter_nans:
        df = df[df[murcko_col].notna()]
        shape2 = df.shape
        if INTERACTIVE:
            info(df, "murcko_from_smiles", f"{shape1[0] - shape2[0]:4d} rows removed.")
    return df


def sss(
    df: pd.DataFrame,
    query: str,
    smiles_col="Smiles",
    is_smarts=False,
    add_h: Union[str, bool] = False,
) -> pd.DataFrame:
    """Substructure search on a Smiles column of the DataFrame.

    Parameters:
    ===========
    df: pd.DataFrame
        The dataframe to apply the function to.
    query: str
        The query Smiles or Smarts.
    smiles_col: str
        The name of the column containing the Smiles strings.
    is_smarts: bool
        Whether the query is a Smarts string (default: False, meaning: query is a Smiles string).
    add_h: bool or str
        Whether to add explicit hydrogens to the molecules before the search.
        Possible values: False, True, "auto".
        If "auto" is used, hydrogens are added only if the query contains explicit hydrogens.
        Default: False.

    Returns a new DF with only the matches."""

    def _sss(smi, qm):
        try:
            m = Chem.MolFromSmiles(smi)
        except:
            return False
        if m is None:
            return False
        if add_h is True:
            m = Chem.AddHs(m)
        if m.HasSubstructMatch(qm):
            return True
        return False

    if add_h == "auto":
        if "#1" in query:
            add_h = True
    if is_smarts:
        q = Chem.MolFromSmarts(query)
    else:
        q = Chem.MolFromSmiles(query)
    df = df.copy()
    if TQDM and len(df) > MIN_NUM_RECS_PROGRESS:
        df["Found"] = df[smiles_col].progress_apply(lambda x: _sss(x, q))
    else:
        df["Found"] = df[smiles_col].apply(lambda x: _sss(x, q))
    df = df[df["Found"]]
    df.drop("Found", axis=1, inplace=True)
    if INTERACTIVE:
        info(df, "sss")
    return df


def sim_search(
    df: pd.DataFrame, query: str, cutoff: float, fp_col="FP", fp_type="ECFC4"
) -> pd.DataFrame:
    """Tanimoto similarity search on a Fingerprint column (default: `FP`) of the DataFrame.
    Returns a new DF with only the matches."""

    def _sss(smi, qm):
        try:
            m = Chem.MolFromSmiles(smi)
        except:
            return False
        if m is None:
            return False
        if m.HasSubstructMatch(qm):
            return True
        return False

    assert fp_type in FPDICT, f"Unknown fingerprint type: {fp_type}."
    assert 0.0 <= cutoff <= 1.0, "Cutoff must be between 0 and 1"

    if isinstance(query, str):
        qmol = smiles_to_mol(query)
    else:  # Assume query is already a mol
        qmol = query
    if qmol is np.nan:
        fp = np.nan
    else:
        fp = FPDICT[fp_type](qmol)

    assert fp is not np.nan, "Query must be a valid SMILES"
    df = df.copy()
    if TQDM and len(df) > MIN_NUM_RECS_PROGRESS:
        df["Sim"] = df[fp_col].progress_apply(
            lambda x: DataStructs.TanimotoSimilarity(x, fp)
        )
    else:
        df["Sim"] = df[fp_col].apply(lambda x: DataStructs.TanimotoSimilarity(x, fp))
    df = df[df["Sim"] >= cutoff]
    df = df.sort_values("Sim", ascending=False)
    if INTERACTIVE:
        info(df, "sim_search")
    return df


def write_sdf(
    df: pd.DataFrame, output_sdf: str, smiles_col="Smiles", keep_smiles=False
):
    """Save the dataframe as an SD file. The molecules are generated from the `smiles_col`.
    Failed molecules are written as NoStructs ("*")."""
    writer = Chem.SDWriter(output_sdf)
    fields = []
    for f in df.keys():
        # if (f != smiles_col) and f != "Mol":
        if f == smiles_col and not keep_smiles:
            continue
        if f == "Mol":
            continue
        fields.append(f)
    for _, rec in df.iterrows():
        mol = smiles_to_mol(rec[smiles_col])
        if mol is np.nan:
            mol = smiles_to_mol("*")
        for f in fields:
            # if f in rec and rec[f]:
            #     mol.SetProp(f, str(rec[f]))
            # else:
            #     mol.SetProp(f, "")
            mol.SetProp(f, str(rec.get(f, "")))
        writer.write(mol)
    writer.close()


def lp(obj, label: str = None, lpad=INFO_WIDTH, rpad=7):
    """log-printing for different kind of objects"""
    if label is not None:
        label_str = label
    if isinstance(obj, str):
        if label is None:
            label_str = "String"
        print(f"{label_str:{lpad}s}: {obj:>{rpad}s}")
        return

    try:
        shape = obj.shape
        if label is None:
            label_str = "Shape"
        else:
            label_str = f"Shape {label}"
        key_str = ""
        has_nan_str = ""
        try:
            keys = list(obj.columns)
            if len(keys) <= 5:
                key_str = " [ " + ", ".join(keys) + " ] "
            num_nan_cols = ((~obj.notnull()).sum() > 0).sum()
            if num_nan_cols > 0:  # DF has nans
                has_nan_str = f"( NAN values in {num_nan_cols} col(s) )"
        except AttributeError:
            pass
        print(
            f"{label_str:{lpad}s}: {shape[0]:{rpad}d} / {shape[1]:{4}d} {key_str} {has_nan_str}"
        )
        return
    except (TypeError, AttributeError, IndexError):
        pass

    try:
        shape = obj.data.shape
        if label is None:
            label_str = "Shape"
        else:
            label_str = f"Shape {label}"
        key_str = ""
        try:
            keys = list(obj.data.columns)
            if len(keys) <= 5:
                key_str = " [ " + ", ".join(keys) + " ] "
        except AttributeError:
            pass
        num_nan_cols = ((~obj.data.notnull()).sum() > 0).sum()
        has_nan_str = ""
        if num_nan_cols > 0:  # DF has nans
            has_nan_str = f"( NAN values in {num_nan_cols} col(s) )"
        print(
            f"{label_str:{lpad}s}:   {shape[0]:{rpad}d} / {shape[1]:{4}d} {key_str} {has_nan_str}"
        )
        return
    except (TypeError, AttributeError, IndexError):
        pass

    try:
        fval = float(obj)
        if label is None:
            label_str = "Number"
        if fval == obj:
            print(f"{label_str:{lpad}s}:   {int(obj):{rpad}d}")
        else:
            print(f"{label_str:{lpad}s}:   {obj:{rpad+6}.5f}")
        return
    except (ValueError, TypeError):
        # print("Exception")
        pass

    try:
        length = len(obj)
        if label is None:
            label_str = "Length"
        else:
            label_str = f"Length {label}"
        print(f"{label_str:{lpad}s}:   {length:{rpad}d}")
        return
    except (TypeError, AttributeError):
        pass

    if label is None:
        label_str = "Object"
    print(f"{label_str:{lpad}s}:   {obj}")


def save_list(lst, fn="list.txt"):
    """Save list as text file."""
    with open(fn, "w") as f:
        for line in lst:
            f.write(f"{line}\n")
