"""
Calculation of Principal Moment of Inertia (PMI).

Generation of multiple conformations and averaging by Median.
"""

from typing import List

import numpy as np

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors as rdMolDesc


def gen_3d(mol: Chem.Mol, n_conformers: int) -> (Chem.Mol, List[int]):
    """
    Generate 3D coordinates for a molecule.
    Generates n_conformers of the molecule and performs optimization on them.

    Returns:
        mol [Chem.Mol]: the rdkit molecule with the embedded optimized conformers.
        conv_res [List[int]]: a list of the convergence results (0 indicates success)."""
    mh = Chem.AddHs(mol)
    Chem.EmbedMultipleConfs(mh, numConfs=n_conformers, params=Chem.ETKDGv3())
    # The convergence results of the individual conformers
    # A value >0 indicates NO Convergence:
    conf_res = []
    for conf_id in range(n_conformers):
        res = 10
        ntries = -1
        iters = [100, 300, 1000]
        while res > 0 and ntries < 3:
            ntries += 1
            res = Chem.UFFOptimizeMolecule(mh, confId=conf_id, maxIters=iters[ntries])
        conf_res.append(res)
    return mh, conf_res


def calc_pmi(mol: Chem.Mol, n_conformers: int, avg=3) -> (float, float):
    """Calculates the Principal Moment of Inertia (PMI) of a molecule.
    The values are calculated for each conformer of the molecule `avg` times.
    Conformers that did not converge in the optimization are skipped.

    Returns:
        PMI1, PMI2 [float]: the PMI values for the molecule. Both values are in the rantge (0, 1)."""
    did_not_converge = 0
    pmi1_list = []
    pmi2_list = []
    for _ in range(avg):
        mol, conv_res = gen_3d(mol, n_conformers)
        did_not_converge += len([x for x in conv_res if x > 0])
        for conf_id in range(n_conformers):
            if conv_res[conf_id] > 0:
                continue
            pmis = sorted(
                [
                    rdMolDesc.CalcPMI1(mol, confId=conf_id),
                    rdMolDesc.CalcPMI2(mol, confId=conf_id),
                    rdMolDesc.CalcPMI3(mol, confId=conf_id),
                ]
            )
            pmi1_list.append(pmis[0] / pmis[2])
            pmi2_list.append(pmis[1] / pmis[2])
    if did_not_converge > 0:
        print(f"* {did_not_converge} minimizations did not converge.")
    return np.median(pmi1_list), np.median(pmi2_list)
