#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##################################################
Extract Active Entries from ChEMBL SQLite Database
##################################################

*Created on Tue Feb 02, 2022 by A. Pahl*

Extract active molecule entries from the SQLite version of the ChEMBL database.

Copy the result file to oracle-server with:

    $ scp chembl_29_active_entries_with_smiles.tsv pahl@oracle-server:/mnt/data/PipelinePilot-data/public/users/pahl/chembl/

"""

import sqlite3
import pandas as pd

from jupy_tools import utils as u
from jupy_tools.deglyco import deglycosylate

# from rdkit.Chem.Scaffolds import MurckoScaffold

VERSION = 29


def deglyco_smiles(smi):
    """
    Deglycosylate SMILES string.
    """
    mol = u.smiles_to_mol(smi)
    mol = deglycosylate(mol)
    result = u.mol_to_smiles(mol)
    return result


if __name__ == "__main__":
    query = """
select distinct(md.chembl_id) as Chembl_Id
from docs d,
     molecule_dictionary md,
     compound_records cr
where md.molregno = cr.molregno
  and cr.doc_id = d.doc_id
  and d.journal in ('J. Nat. Prod.', 'J Nat Prod');"""

    print(f"Extracting Natural Product entries from ChEMBL {VERSION} (SQLite)...")
    conn = sqlite3.connect(f"./chembl_{VERSION}.db")
    df = pd.read_sql(query, conn)
    conn.close()
    df.to_csv(f"chembl_{VERSION}_np_entries.tsv", sep="\t", index=False)
    # df = pd.read_csv(f"chembl_{VERSION}_active_entries.tsv", sep="\t", low_memory=False)
    print(f"{df.shape[0]} NP entries extracted.")
    print("Merging Smiles from full nocanon data...")
    df_mc = pd.read_csv(f"./chembl_{VERSION}_np_full_nocanon.tsv", sep="\t")
    df = pd.merge(df, df_mc, how="inner", on="chembl_id")
    df = df.drop("Name", axis=1)
    print(f"{df.shape[0]} entries merged.")
    print("Deglycosylation...")
    df["Smiles_deglyco"] = df["Smiles"].apply(deglyco_smiles)
    df = df[~df["Smiles_deglyco"].isna()]
    df = df.drop("Smiles", axis=1)
    df = df.rename(columns={"Smiles_deglyco": "Smiles"})
    print(f"{df.shape[0]} entries remained after deglycosylation.")
    print("Saving...")
    df.to_csv(f"chembl_{VERSION}_np_full_nocanon_deglyco.tsv", sep="\t", index=False)
    print("Done.")
