#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##################################################
Extract Active Entries from ChEMBL SQLite Database
##################################################

*Created on Tue Feb 02, 2022 by A. Pahl*

Extract active molecule entries from the SQLite version of the ChEMBL database.
The script requires the ChEMBL SQLite database `chembl_{VERSION}.db` to be
present in the current working directory.
"""

import sys

import sqlite3
import pandas as pd

from rdkit.Chem.Scaffolds import MurckoScaffold


if __name__ == "__main__":
    #   and act.standard_type in ('IC50', 'EC50', 'Ki', 'Kd')
    #   and act.standard_units = 'nM'
    #   and act.standard_relation in ('=', '<', '<=')

    query = """
select md.chembl_id, act.pchembl_value as pchembl, act.standard_relation as relation,
       pfc.l1 as pfc_l1, pfc.l2 as pfc_l2, pfc.l3 as pfc_l3, pfc.l4 as pfc_l4, pfc.l5 as pfc_l5
from activities act, molecule_dictionary md,
      assays, target_dictionary td,
      component_class cc, target_components tc,
      protein_family_classification pfc
where act.molregno = md.molregno
  and assays.assay_id = act.assay_id
  and td.tid = assays.tid
  and tc.tid = assays.tid
  and tc.component_id = cc.component_id
  and cc.protein_class_id = pfc.protein_class_id
  and assays.confidence_score = 9
  and assays.assay_type = 'B'
  and td.target_type = 'SINGLE PROTEIN'
  and (act.data_validity_comment is Null  or act.data_validity_comment = 'Manually validated');"""

    print("Extract active entries from ChEMBL.")
    if len(sys.argv) != 3:
        print("Usage: extract_nps_from_sqlite.py <chembl version> <medchem,medchemrac>")
        print("Example: extract_nps_from_sqlite.py 31 medchem")
    VERSION = sys.argv[1]
    MODE = sys.argv[2]
    print(f"Extracting active entries from ChEMBL {VERSION} (SQLite)...")
    conn = sqlite3.connect(f"./chembl_{VERSION}.db")
    df = pd.read_sql(query, conn)
    conn.close()
    df.to_csv(f"chembl_{VERSION}_active_entries.tsv", sep="\t", index=False)
    # df = pd.read_csv(f"chembl_{VERSION}_active_entries.tsv", sep="\t", low_memory=False)
    print(f"{df.shape[0]} active entries extracted.")
    print(f"Merging Smiles from {MODE} data...")
    df_mc = pd.read_csv(f"./chembl_{VERSION}_{MODE}.tsv", sep="\t")
    df = pd.merge(df, df_mc, how="inner", on="chembl_id")
    df = df.drop("Name", axis=1)
    print(f"{df.shape[0]} entries merged.")
    print("Adding Murcko Smiles...")
    df["Murcko_Smiles"] = df["Smiles"].apply(
        MurckoScaffold.MurckoScaffoldSmilesFromSmiles
    )
    print("Saving...")
    df.to_csv(f"chembl_{VERSION}_active_entries_with_smiles.tsv", sep="\t", index=False)
    print("Done.")
