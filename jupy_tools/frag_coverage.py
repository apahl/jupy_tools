#!/usr/bin/env python3
# """Calculate the fragment coverage for a given dataset."""

import sys

import pandas as pd
import numpy as np

# from rdkit import DataStructs
# from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors as Desc

from jupy_tools import utils as u


def fc_test():
    df_frag = pd.DataFrame(
        {"idm": [1, 2, 3], "Smiles": ["C2CCN1CCC1C2", "C1CCCCC1", "C1CCNCC1"]}
    )
    df_frag["Mol"] = df_frag["Smiles"].apply(u.smiles_to_mol)

    ds = pd.DataFrame({"cid": [1], "Smiles": ["C3CCC12CCCC(C1)N2C3"]})
    df_res = frag_coverage(ds, "cid", "Smiles", df_frag)
    df_exp = pd.DataFrame(
        {
            "cid": [1],
            "n_frags_exp": [2],
            "n_atoms_covered_exp": [11],
            "n_atoms_mol_exp": [11],
        }
    )
    df_test = pd.merge(df_res, df_exp, on="cid", how="left")
    assert df_test["n_frags"].equals(df_test["n_frags_exp"])
    assert df_test["n_atoms_covered"].equals(df_test["n_atoms_covered_exp"])
    assert df_test["n_atoms_mol"].equals(df_test["n_atoms_mol_exp"])
    return df_test


def frag_coverage(
    ds: pd.DataFrame, id_col: str, smiles_col: str, df_frag: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the fragment coverage for a given dataset.
    The fragments are used in descending size order to avoid double counting.
    A fragment is only added to the coverage if it adds atoms to the total count of covered atoms.
    So, if indole was already found, pyrrole is not added to the coverage.

    Parameters
    ==========
    ds : pd.DataFrame
        Dataset to calculate the fragment coverage for.
        The dataset must contain a column named 'Smiles' and an `id_col` with the identifiers.
    id_col: the column containing the identifiers for the molecules in the dataset.
    smiles_col: the column containing the SMILES for the molecules in the dataset.
    df_frag : pd.DataFrame
        DataFrame containing the fragments to calculate the coverage for.
        The DataFrame must contain a column named 'Mol' that contains the fragment molecules
        and a unique 'idm' column containing the identifier.

    Returns
    =======
    pd.DataFrame
        DataFrame containing the fragment coverage for each molecule in the dataset.
        The function adds 5 columns to the input dataset:
        'n_frags': number of fragments found in the molecule
        'frag_ids': list of fragment ids found in the molecule
        'n_atoms_covered': number of atoms covered by the fragments in the molecule
        'n_atoms_mol': number of atoms in the molecule
        'frag_coverage': fraction of atoms covered by the fragments in the molecule
    """

    # Add the number of heavy atoms to the fragments for size sorting
    df_frag["n_atoms"] = df_frag["Mol"].apply(Desc.HeavyAtomCount)
    df_frag = df_frag.sort_values("n_atoms", ascending=False).reset_index(drop=True)

    # Initialize the results fields
    c_ids = []
    n_frags = []
    frag_ids = []
    n_atoms_covered = []
    n_atoms = []
    frag_cvrg = []

    ctr = 0
    num_recs = len(ds)
    report_every = 1000
    if num_recs <= report_every:
        report_every = 100
    for _, mol_rec in ds.iterrows():
        ctr += 1
        if ctr % report_every == 0:
            print(f"Processed {ctr:7d} of {num_recs:7d} records...  ", end="\r")
            sys.stdout.flush()
        mol = u.smiles_to_mol(mol_rec[smiles_col])
        if mol is np.nan:
            continue

        hit_atoms = set()
        mol_n_atoms = Desc.HeavyAtomCount(mol)
        mol_frag_ids = []
        for _, frag_rec in df_frag.iterrows():
            ssm = mol.GetSubstructMatches(frag_rec["Mol"])
            if len(ssm) > 0:
                for match in ssm:
                    prev_len = len(hit_atoms)
                    hit_atoms.update(match)
                    if len(hit_atoms) > prev_len:
                        mol_frag_ids.append(frag_rec["idm"])
        c_ids.append(mol_rec[id_col])
        n_frags.append(len(mol_frag_ids))
        frag_ids.append(", ".join(map(str, mol_frag_ids)))
        n_atoms_covered.append(len(hit_atoms))
        n_atoms.append(mol_n_atoms)
        frag_cvrg.append(round(len(hit_atoms) / mol_n_atoms, 3))

    print(f"Processed {ctr:7d} of {num_recs:7d} records...  ")
    print("Creating result...")
    sys.stdout.flush()

    df_res = pd.DataFrame(
        {
            id_col: c_ids,
            "n_frags": n_frags,
            "frag_ids": frag_ids,
            "n_atoms_covered": n_atoms_covered,
            "n_atoms_mol": n_atoms,
            "frag_coverage": frag_cvrg,
        }
    )

    df_final = pd.merge(ds, df_res, on=id_col, how="left")
    print("Done.")
    return df_final
