#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#######
Cluster
#######

*Ported from `mol_frame` on June 2nd, 2025 by A. Pahl*

Tools for clustering structures.

The port replaces the MolFrame class by a classical Pandas DataFrame.
"""

import json

import pandas as pd

from rdkit import DataStructs

# from rdkit.Chem.Descriptors import qed

# from mol_frame import mol_frame as mf, templ as mfht, viewers as mfv, tools as mft
from jupy_tools import utils as u, mol_view as mv, templ

# from typing import List,
from typing import Dict, List, Tuple  # , Union


class Cluster:
    """Container class for clustering process.
    Operates on a copy of the original DataFrame."""

    def __init__(self, df: pd.DataFrame, config: Dict[str, str] = {}):
        """
        Parameters:
            df: Pandas DataFrame instance.
            config defaults:
                ID_COL: The column with the compound identifiers, e.g. "Compound_Id".
                    default: "Compound_Id"
                FP: The fingerprint function that is called on the molecule, eg. ``mf.FPDICT["fcfc6"]``.
                    default: "fcfc6"
                METRIC: The method for comparing the fingerprints, e.g. ``Tanimoto`` or ``Cosine``.
                    Methods known by name are: ``Tanimoto`` (default), ``Cosine`` and ``Dice``. These can be given as string.
                    Other methods can be passed as callable object.
                    default: "Tanimoto"
                CUTOFF: Similarity cutoff between 0 and 1.
                ADD_ACT_COL: Whether to add the minimum activity to each cluster, default is True.
                ACT_COL: The column to use for the minimum activity (e.g. ``Xyz_IC50_uM``).
                    Required when ``add_min_act`` is True.
                MIN_ACT_COL: The name of column for the minimum activity.
                    If not given, it will be the name of the ``act_col`` with ``Cluster_Min_`` prefixed.
                MERGE_CPDS: Option for manual curation of the clusters.
                    A list of lists of ``id_col`` compound identifiers. The compounds in each sublist are assigned to the same cluster as the first compound in each sublist.
                MERGE_CLUSTERS: Option for manual curation of the clusters.
                    A list of lists of ``id_col`` compound identifiers. The clusters of the compounds in each sublist are merged into one cluster (that of the first compound in the list).
                RESCUE_FACTOR, MARK_SIM_FACTOR,
                NAME: base file name of the project."""

        self.molf = df.copy()
        self.config = {
            "FP": "ECFP6",
            "ID_COL": "Compound_Id",
            "METRIC": "Tanimoto",
            "CUTOFF": 0.55,
            "ADD_MIN_ACT": False,
            "ACT_COL": None,
            "MIN_ACT_COL": None,
            "MERGE_CPDS": None,
            "MERGE_CLUSTERS": None,
            "RESCUE_SINGLETONS": True,
            "RESCUE_FACTOR": 0.85,
            "MARK_SIM_FACTOR": 0.66,
            "NAME": "clustering",
        }
        self.config.update(config)
        self.id_col = self.config["ID_COL"]
        self._fp = self.config["FP"]

    def __str__(self):
        shape = self.molf.shape
        keys = list(self.molf.keys())
        return f"MolFrame  Rows: {shape[0]:6d}  Columns: {shape[1]:2d}   {keys}"

    def __repr__(self):
        return self.__str__()

    def write(self, **kwargs):
        bn = kwargs.get("name", self.config["NAME"])
        self.molf.to_csv(f"{bn}.tsv", sep="\t", index=False)
        self.cl_info.to_csv(f"{bn}_info.tsv", sep="\t", index=False)
        with open(f"{bn}_config.json", "w") as f:
            json.dump(self.config, f)

    def copy(self):
        result = Cluster(self.molf, self.config)
        result._fp = self._fp
        result._fp_sim = self._fp_sim
        result.cl_info = self.cl_info
        return result

    def report(
        self,
        columns: List[str] = ["Compound_Id", "Smiles"],
        title: str = "Cluster Report",
        intro: str = "Clusters in their order in the dataset.",
        summary: str = "",
        truncate=100,
        replace_nans=True
    ):
        """Write a HTML report. `Cluster_No` has to be present in the DataFrame.
        Writes the clusters in the order as they are in the dataset.

        Arguments:
            columns: List of columns to include.
            title: The report title.
            intro: Some text used for introduction of the report.
            summary: text displayed at the end of the report. By default the configuration will be shown.
        """

        def add_cluster(cl_no):
            sim_cluster = cl_info.query("Cluster_No == @cl_no")["Similar_To"].values[0]
            if sim_cluster > 0:
                sim_text = (
                    f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Similar to Cluster {sim_cluster})"
                )
            else:
                sim_text = ""
            mf_cl = df.query("Cluster_No == @cl_no")[columns].copy()
            mf_cl = u.add_mol_col(mf_cl)
            html.append(
                f"<br><h2>Cluster {cl_no} ({len(mf_cl)} Members){sim_text}</h2><br>"
            )
            grid = mv.mol_grid(mf_cl, id_col=id_col, mols_per_row=6, as_html=False, truncate=truncate)
            html.append(grid)

        id_col = self.id_col
        cl_info = self.cl_info

        if "Cluster_No" not in columns:
            columns = ["Cluster_No"] + columns
        if id_col not in columns:
            columns = [id_col] + columns
        if "Smiles" not in columns:
            columns.append("Smiles")
        df = self.molf[columns].copy()
        if replace_nans:
            df = df.fillna("")        

        if summary == "":
            summary = f"<p><b>Configuration:</b> {str(self.config)}</p>"

        html = [f"<h1>{title}</h1><br>{intro}<br><br>"]

        clusters = df["Cluster_No"].unique()
        for cl_no in clusters:
            add_cluster(cl_no)

        html.append(summary)

        templ.write(templ.page("\n".join(html)), "Clusters.html")

    def curate(self):
        """Apply manual curations."""
        cutoff = self.config["CUTOFF"]
        add_min_act = self.config["ADD_MIN_ACT"]
        act_col = self.config["ACT_COL"]
        min_act_col = self.config["MIN_ACT_COL"]
        merge_cpds = self.config["MERGE_CPDS"]
        merge_clusters = self.config["MERGE_CLUSTERS"]
        mark_sim_factor = self.config["MARK_SIM_FACTOR"]
        molf = self.molf.copy()
        print("* curating...")

        if merge_cpds is not None:
            print("* merging compounds...")
            # Iterate over the sublists
            for cpd_list in merge_cpds:
                # Get the Cluster_No of the first compound
                cl_no = int(
                    molf[molf[self.id_col] == cpd_list[0]][
                        "Cluster_No"
                    ].values[0]
                )
                # Assign this Cluster_No to the remaining compounds in the list:
                idx = molf[molf[self.id_col].isin(cpd_list[1:])].index
                molf.loc[idx, "Cluster_No"] = cl_no

        if merge_clusters is not None:
            print("* merging clusters...")
            # Iterate over the sublists
            for cpd_list in merge_clusters:
                # Get the Cluster_No of the first compound
                cl_no = int(
                    molf[molf[self.id_col] == cpd_list[0]][
                        "Cluster_No"
                    ].values[0]
                )
                # Assign this Cluster_No to the clusters the remaining compounds in the list are in:
                for cpd_id in cpd_list[1:]:
                    # Get the Cluster_No of the other compound
                    cl_no_old = int(
                        molf[molf[self.id_col] == cpd_id][
                            "Cluster_No"
                        ].values[0]
                    )
                    # Assign the new Cluster_No to all cpds with this old Cluster_No
                    idx = molf[molf["Cluster_No"] == cl_no_old].index
                    molf.loc[idx, "Cluster_No"] = cl_no

        # Finalize Cluster property assignments
        # Assign Cluster_Size
        tmp = molf.copy()
        tmp = (
            tmp.groupby(by="Cluster_No")
            .count()
            .reset_index()
            .rename(columns={"Smiles": "Cluster_Size"})[["Cluster_No", "Cluster_Size"]]
        )
        molf = pd.merge(molf, tmp, on="Cluster_No", how="inner")

        if add_min_act:
            # Rescuing and manual curation might mess up the cluster ordering
            # Renumber and reorder the clusters by activity.
            print("* re-assigning cluster numbers")
            molf = molf.sort_values(act_col, ascending=False)
            # But keep the singletons at the end
            molf = pd.concat(
                [
                    molf.query("Cluster_Size > 1"),
                    molf.query("Cluster_Size == 1"),
                ]
            ).reset_index(drop=True)
            tmp = molf.drop_duplicates(subset=["Cluster_No"]).copy()
            tmp = tmp.reset_index(drop=True)
            # We need a temporary 2nd column for the Cluster_No
            molf["Cluster_No_old"] = molf["Cluster_No"]
            for cl_no, (_, rec) in enumerate(tmp.iterrows(), 1):
                cl_no_old = rec["Cluster_No"]
                idx = molf[molf["Cluster_No_old"] == cl_no_old].index
                molf.loc[idx, "Cluster_No"] = cl_no
            # Remove the temp. column again
            molf.drop("Cluster_No_old", axis=1, inplace=True)

        cl_info = molf.drop_duplicates(subset=["Cluster_No"]).copy()
        # cl_info = pd.merge(cl_info, tmp, on="Cluster_No", how="inner")

        # Mark similar clusters
        print("* searching for similar clusters")
        cl_info = cl_info.sort_values("Cluster_No")
        tmp = cl_info.copy()
        tmp = u.add_fps(tmp, fp_type=self._fp)
        fps = tmp["FP"].to_dict()  # link indexes and fingerprints

        cl_info["Similar_To"] = 0
        assigned = set()
        mark_sim_cutoff = cutoff * mark_sim_factor
        idx_list = list(cl_info.index)
        mark_sim = {}
        for ix, idx1 in enumerate(idx_list):
            # if idx1 in assigned:
            #     continue
            for idx2 in idx_list[ix + 1 :]:
                if idx2 in assigned:
                    continue

                if self._fp_sim(fps[idx1], fps[idx2]) < mark_sim_cutoff:
                    continue
                mark_sim[idx2] = idx1
                assigned.add(idx2)

        for idx in mark_sim:
            cl_no_simto = cl_info.loc[mark_sim[idx], "Cluster_No"]
            # cl_no = cl_info.ix[idx]["Cluster_No"]
            # print(idx, mark_sim[idx], cl_no, cl_no_simto)
            cl_info.loc[idx, "Similar_To"] = cl_no_simto

        cl_info = cl_info.reset_index(drop=True)

        # Assign Minimum Activity per cluster, if given
        if add_min_act:
            tmp = molf.copy()
            tmp = (
                tmp.groupby(by="Cluster_No")
                .min()
                .reset_index()
                .rename(columns={act_col: min_act_col})[["Cluster_No", min_act_col]]
            )

            cl_info = pd.merge(cl_info, tmp, on="Cluster_No", how="inner")
            molf = pd.merge(molf, tmp, on="Cluster_No", how="inner")
            molf = molf.sort_values(
                ["Cluster_No", act_col], ascending=[True, True]
            )
        columns = ["Cluster_No", "Cluster_Size", "Similar_To"]
        if add_min_act:
            columns.append(min_act_col)
        cl_info = cl_info[columns].sort_values("Cluster_No")
        self.molf = molf.copy()
        self.cl_info = cl_info.copy()

    def cluster_eager(self, verbose=True):
        """Eager Clustering. Includes Curating (see ``curate`` function).
        The clusters are generated in the order, in which the compounds are in the DataFrame.
        The first compound defines the first cluster. All remaining compounds in the DataFrame
        that have a similarity >= the given cutoff are added to that cluster and removed from the set.
        If no further compounds can be added to a cluster, the next available compound in the DataFrame starts the next cluster.
        In a 2nd round, singletons are rescued and assigned to existing clusters, when their similarity is >= ``rescue_factor`` * cutoff."""

        molf = self.molf.copy()
        metric = self.config["METRIC"]
        cutoff = self.config["CUTOFF"]
        add_min_act = self.config["ADD_MIN_ACT"]
        act_col = self.config["ACT_COL"]
        min_act_col = self.config["MIN_ACT_COL"]
        rescue_factor = self.config["RESCUE_FACTOR"]

        if isinstance(metric, str):
            if "tani" in metric.lower():
                self._fp_sim = lambda x, y: DataStructs.TanimotoSimilarity(x, y)
            elif "cos" in metric.lower():
                self._fp_sim = lambda x, y: DataStructs.CosineSimilarity(x, y)
            elif "dice" in metric.lower():
                self._fp_sim = lambda x, y: DataStructs.DiceSimilarity(x, y)
            else:
                raise ValueError(
                    f"Unknown similarity method {metric}."
                    "Please use one of ``Tanimoto``, ``Cosine`` or ``Dice``"
                )

        if add_min_act:
            if not isinstance(act_col, str):
                raise ValueError(
                    "act_col is required when add_min_act is True (which it is)."
                )
            if not isinstance(min_act_col, str):
                min_act_col = f"Cluster_Min_{act_col}"
            molf = molf.sort_values(act_col, ascending=True)

        print("* adding fingerprints...")
        molf = molf.reset_index(drop=True)
        molf = u.add_fps(molf, fp_type=self._fp)
        fps = molf["FP"].to_dict()  # link indexes and fingerprints
        molf.drop("FP", axis=1, inplace=True)
        print("* eager clustering...")
        clusters = []
        singletons = []
        assigned = set()
        idx_list = list(fps.keys())
        for ix1, idx1 in enumerate(idx_list):
            if idx1 in assigned:
                continue
            new_cluster = [idx1]
            assigned.add(idx1)
            for idx2 in idx_list[ix1 + 1 :]:
                if idx2 in assigned:
                    continue
                if self._fp_sim(fps[idx1], fps[idx2]) < cutoff:
                    continue
                new_cluster.append(idx2)
                assigned.add(idx2)
            if len(new_cluster) == 1:
                singletons.append(new_cluster[0])
            else:
                clusters.append(new_cluster)

        single_len = len(singletons)
        print(f"  {len(clusters)} clusters and {single_len} singletons were assigned.")
        if self.config["RESCUE_SINGLETONS"]:
            print("* rescue singletons...")
            assigned = set()
            rescue_cutoff = cutoff * rescue_factor  # reduce the cutoff
            clusters_new = []
            for cl in clusters:
                cl_new = cl.copy()
                for idx1 in cl:
                    cl_new.append(idx1)
                    for idx2 in singletons:
                        if idx2 in assigned:
                            continue
                        if self._fp_sim(fps[idx1], fps[idx2]) < rescue_cutoff:
                            continue
                        cl_new.append(idx2)
                        assigned.add(idx2)
                clusters_new.append(cl_new)
            clusters = clusters_new
            print(f"  {len(assigned)} singletons were rescued.")
            for idx in singletons:
                if idx not in assigned:
                    clusters.append([idx])

            molf["Rescued"] = False
            molf.loc[list(assigned), "Rescued"] = True

        if verbose:
            print("\nClusters:")
            print(clusters, "\n")

        print("* assigning clusters to MolFrame...")
        molf["Cluster_No"] = 0
        for cl_no, cl in enumerate(clusters, 1):
            molf.loc[cl, "Cluster_No"] = cl_no

        self.molf = molf.copy()
        self.curate()


def read(self, name: str) -> Cluster:
    bn = name
    molf = pd.read_csv(f"{bn}.tsv", sep="\t")
    result = Cluster(molf)
    cl_info = pd.read_csv(f"{bn}_info.tsv", sep="\t")
    result.cl_info = cl_info
    with open(f"{bn}_config.json", "r") as f:
        result.config = json.load(f)
    return result
