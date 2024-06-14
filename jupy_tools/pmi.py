"""
Calculation of Principal Moment of Inertia (PMI).

Generation of multiple conformations and averaging by Median.
"""

from typing import List

import numpy as np

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors as rdMolDesc
from typing import Tuple


def get_stereo_counts(mol) -> Tuple[int, int, bool]:
    """Count the number of specified and unspecified atom stereo centers.
    Only chirality at carbons is considered.
    Returns a tuple of (num_specified: int, num_specified: int, is_diastereomer: bool) counts.
    A molecule is a potential diastereomer when it has either
    - no specified stereo centers and more than one unspecified stereo center, or
    - at least one specified stereo center and one or more unspecified stereo centers."""
    num_spec = 0
    num_unspec = 0
    atoms = mol.GetAtoms()
    chiraltags = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True)
    # example output [(1, '?'), (2, 'R'), (5, 'S'), (8, 'S')]
    for tag in chiraltags:
        if tag[1] == "R" or tag[1] == "S":
            num_spec += 1
        else:
            if atoms[tag[0]].GetAtomicNum() == 6:
                num_unspec += 1
    if num_spec == 0 and num_unspec > 1:
        return num_spec, num_unspec, True
    if num_spec > 0 and num_unspec > 0:
        return num_spec, num_unspec, True
    return num_spec, num_unspec, False


def gen_3d(mol: Chem.Mol, n_conformers: int) -> Tuple[Chem.Mol, List[int]]:
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


def calc_pmi(mol: Chem.Mol, n_conformers: int, avg=3) -> Tuple[float, float]:
    """Calculates the Principal Moment of Inertia (PMI) of a molecule.
    The values are calculated for each conformer of the molecule `avg` times.
    Conformers that did not converge in the optimization are skipped.

    Returns:
        PMI1, PMI2 [float]: the PMI values for the molecule. PMI1 is in the range (0, 1), PMI2 in range (0.5, 1)."""
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
    return round(np.median(pmi1_list), 3), round(np.median(pmi2_list), 3)
