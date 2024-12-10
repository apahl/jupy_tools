#!/usr/bin/env python3
"""Using a Minimum Spanning Tree for compound clustering.
The resulting tree displays the activity of the compounds in a continuous color scale.
Module requirements: RDKit, NetworkX, pygraphviz, holoviews and Pandas."""

import pandas as pd

from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdFingerprintGenerator

import networkx as nx

from jupy_tools import utils as u, mol_view as mv

NBITS = 2048
FPDICT = {}

EFP4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)
EFP6 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=NBITS)
FFP4 = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=NBITS,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)
FFP6 = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, fpSize=NBITS,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)

FPDICT["ECFC4"] = lambda m: EFP4.GetCountFingerprint(m)
FPDICT["ECFC6"] = lambda m: EFP6.GetCountFingerprint(m)
FPDICT["ECFP4"] = lambda m: EFP4.GetFingerprint(m)
FPDICT["ECFP6"] = lambda m: EFP6.GetFingerprint(m)    
FPDICT["FCFP4"] = lambda m: FFP4.GetFingerprint(m)
FPDICT["FCFP6"] = lambda m: FFP6.GetFingerprint(m)    


class ClusterMST:
    """ClusterMST class.
    Structures are expected as Smiles."""
    def __init__(
            self, df, id_col, act_col, top_n_act: int, reverse=False, sim_cutoff=0.6, num_sim=10, smiles_col="Smiles", fp="ECFC4"
        ):
        """Initialize the class.
        
        Parameters:
        ===========
        df: Pandas DataFrame with the data.
        id_col: ID column name.
        act_col: Activity column name.
        reverse: Reverse the activity values for the color scale.
        sim_cutoff: Similarity cutoff for the Minimum Spanning Tree.
        smiles_col: Smiles column name."""
        
        self.id_col = id_col
        self.act_col = act_col
        self.top_n_act = top_n_act
        self.reverse = reverse
        # After the processing, self.df will also contain the graph coordinates
        self.df = df.copy()
        self.df = self.df.sort_values(self.act_col, ascending=self.reverse).reset_index(drop=True)
        self.sim_cutoff = sim_cutoff
        self.num_sim = num_sim
        self.smiles_col = smiles_col
        self.fp_method = fp
        self.fps = None
        self.ids = self.df[self.id_col].tolist()
        self.mst = None  # MST Graph
        self.edges = None  # the MST connections as DataFrame
        
    def _calc_fps(self):
        """Calculate the selected fingerprints."""
        # self.fps = [Chem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2) for smi in self.df[self.smiles_col]]
        fp_call = FPDICT[self.fp_method]
        self.fps = {cpd_id: fp_call(Chem.MolFromSmiles(smi)) for cpd_id, smi in self.df[[self.id_col, self.smiles_col]].values}
        if len(self.ids) != len(self.fps):
            raise ValueError("The number of IDs and fingerprints does not match.")
    
    def calc_mst(self):
        """Calculate the Minimum Spanning Tree."""

        self._calc_fps()
        top_n_ids = self.df.head(self.top_n_act)[self.id_col].copy().tolist()
        sims_full = []
        for id1 in top_n_ids:
            for id2 in self.fps.keys():
                if id1 == id2:
                    continue
                sim = DataStructs.TanimotoSimilarity(self.fps[id1], self.fps[id2])
                if sim >= self.sim_cutoff:
                    # Similarity both ways:
                    sims_full.append((id1, id2, sim))
                    # sims_full.append((self.ids[idx2], self.ids[idx1], sim))
                
        df_sims = pd.DataFrame(sims_full, columns=[self.id_col+"_1", self.id_col+"_2", "Similarity"])
        # df_sims = df_sims[df_sims["Similarity"] >= self.sim_cutoff]
        df_sims = df_sims.sort_values([f"{self.id_col}_1", "Similarity"], ascending=[True, False]).reset_index(drop=True)
        # id_set = set()
        # for idx, rec in self.df.iterrows():
        #     if idx >= self.top_n_act - 1:
        #         break
        #     tmp = df_sims[df_sims[self.id_col+"_1"] == rec[self.id_col]].head(self.num_sim).copy()
        #     id_set.add(rec[self.id_col])
        #     id_set.update(tmp[self.id_col+"_2"].tolist())
        
        # Keep the top `num_sim` similar compounds for each compound:
        df_sims = df_sims.groupby(f"{self.id_col}_1").head(self.num_sim).reset_index(drop=True)
        id_set = set(top_n_ids)
        id_set.update(df_sims[self.id_col+"_2"].tolist())
        
        self.df = self.df[self.df[self.id_col].isin(id_set)].copy()
        self.ids = self.df[self.id_col].tolist()
        self._calc_fps()
        g = nx.Graph()
        entries = set()
        fps_len = len(self.fps)
        for id1 in self.fps.keys():
            for id2 in self.fps.keys():
                if id1 == id2:
                    continue
                sim = DataStructs.TanimotoSimilarity(self.fps[id1], self.fps[id2])
                g.add_edge(id1, id2, weight=1+9*(1-sim), len=1+9*(1-sim))
                entries.add(id1)
                entries.add(id2)
        self.mst = nx.minimum_spanning_tree(g)
        self.pos = nx.planar_layout(self.mst)
        self.pos = nx.kamada_kawai_layout(self.mst, pos=self.pos)
        mst_points = pd.DataFrame.from_dict(self.pos, orient='index', columns=['X', 'Y']).reset_index().rename(columns={'index': self.id_col})
        self.df = pd.merge(self.df, mst_points, on=self.id_col, how="inner")
        edges = []
        for (cpd1, cpd2) in self.mst.edges():
            edge = (self.pos[cpd1][0], self.pos[cpd1][1], self.pos[cpd2][0], self.pos[cpd2][1])
            edges.append(edge)
        self.edges = pd.DataFrame(edges, columns=["X1", "Y1", "X2", "Y2"])
        print(f"MST generated with {len(entries)} entries.")
