"""Deglycosylation functionality, developed by Jose-Manuel Gally (project NPFC):
https://github.com/mpimp-comas/npfc"""

from typing import Union

from collections import Counter

from networkx import Graph

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Mol
import rdkit.Chem.Descriptors as Descriptors


def fuse_rings(rings: tuple) -> list:
    """
    Check for atom indices in common between rings to aggregate them into fused rings.
    :param rings: the ring atoms as provided by the RDKit function mol.GetRingInfo().AtomRings() (iteratble of iteratable of atom indices).
    :return: the fused ring atoms (list of lists of atom indices)
    """
    # condition to exit from recursive fusion of ring atom indices
    done = False

    while not done:
        fused_rings = []
        num_rings = len(rings)
        # pairwise check for common atoms between rings
        for i in range(num_rings):
            # define a core
            fused_ring = set(rings[i])
            for j in range(i + 1, num_rings):
                # detect if ring is fused
                if set(rings[i]) & set(rings[j]):
                    # add fused ring to our rign atom list
                    fused_ring = fused_ring.union(rings[j])
            # either lone or fused ring, first check if not already in fused_rings
            if any([fused_ring.issubset(x) for x in fused_rings]):
                continue
            fused_rings.append(list(fused_ring))
        rings = list(fused_rings)
        # there are no rings to fuse anymore
        if num_rings == len(rings):
            done = True

    return rings


def _is_sugar_like(ring_aidx: list, mol: Mol):
    """Indicate whether a ring (defined by its atom indices) in a molecule is sugar-like or not.
    Several conditions are to be met for a ring to be considered sugar-like:
        1. size: either 5 or 6 atoms
        2. elements: 1 oxygen and the rest carbons
        3. hybridization: ring atoms need have single bonds only
        4. connection points (next to the ring oxygen): at least 1 has an oxygen as neighbor
        5. subsituents (not next tot the ring oxygen): at least 1/2 (for 5-6-membered rings) have an oxygen as neighbor
    :param ring_aidx: the molecule indices of the ring to investigate
    :param mol: the molecule that contain the ring
    :return: True if the ring complies to the 5 conditions above, False otherwise.
    """
    # ring size: only 5-6 membered rings, rings are already fused when this function is called
    ring_size = len(ring_aidx)
    if ring_size != 5 and ring_size != 6:
        return False

    # access the actual atom objects quickier
    ring_atoms = [
        mol.GetAtomWithIdx(x) for x in ring_aidx
    ]  # ring atoms are in the same order as ring_aidx

    # atom composition
    elements = [x.GetAtomicNum() for x in ring_atoms]
    element_counter = Counter(elements)
    if not (
        (ring_size == 5 and element_counter[6] == 4 and element_counter[8] == 1)
        or (ring_size == 6 and element_counter[6] == 5 and element_counter[8] == 1)
    ):
        return False

    # hybridization of carbon atoms (check if only single bonds attached to the ring)
    carbon_atoms = [x for x in ring_atoms if x.GetAtomicNum() == 6]
    if any(
        [x for x in carbon_atoms if x.GetHybridization() != 4]
    ):  # to check if no H attached in case of the * position
        return False

    # to define connection points and substituents, we first need to identify the position of the ring oxygen
    oxygen_aidx = [x for x in ring_atoms if x not in carbon_atoms][
        0
    ].GetIdx()  # only 1 oxygen in ring

    # connection points: 1 need at least 1 oxygen as neighbor
    cps = []
    cps_ok = False
    for carbon_atom in carbon_atoms:
        neighbors = carbon_atom.GetNeighbors()
        # if the ring oxygen is next to this atom, this atom is a connection point
        if any([n.GetIdx() == oxygen_aidx for n in neighbors]):
            cps.append(carbon_atom)
            # at least 1 of the connection points has to have an oxygen as side chain
            if any(
                [n.GetAtomicNum() == 8 and n.GetIdx() != oxygen_aidx for n in neighbors]
            ):
                cps_ok = True
    if not cps_ok:
        return False

    # substituents
    substituents = [
        x for x in carbon_atoms if x.GetIdx() not in [y.GetIdx() for y in cps]
    ]
    count_oxygens = 0
    for substituent in substituents:
        side_chain_atoms = [
            x for x in substituent.GetNeighbors() if x.GetIdx() not in ring_aidx
        ]
        if len(side_chain_atoms) > 0:
            if (
                not side_chain_atoms[0].GetAtomicNum() == 8
            ):  # do not check for the degree here because there are connections on substituents too!
                return False
            count_oxygens += 1
    # at least 1 oxygen for 5-membered rigns and 2 for 6-membered rings
    if (ring_size == 6 and count_oxygens < 2) or (ring_size == 5 and count_oxygens < 1):
        return False

    return True


def deglycosylate(mol: Mol, mode: str = "run") -> Union[Mol, Graph]:
    """Function to deglycosylate molecules.

    Several rules are applied for removing Sugar-Like Rings (SLRs) from molecules:

        1. Only external SLRs are removed, so a molecule with aglycan-SLR-aglycan is not modified
        2. Only molecules with both aglycans and SLRs are modified (so only SLRs or none are left untouched)
        3. Linear aglycans are considered to be part of linkers and are thus never returned as results
        4. Glycosidic bonds are defined as either O or CO and can be linked to larger linear linker. So from a SLR side, either nothing or only 1 C are allowed before the glycosidic bond oxygen
        5. Linker atoms until the glycosidic bond oxygen atom are appended to the definition of the SLR, so that any extra methyl is also removed.


    .. image:: _images/std_deglyco_algo.svg
        :align: center

    :param mol: the input molecule
    :param mode: either 'run' for actually deglycosylating the molecule or 'graph' for returning a graph of rings instead (useful for presentations or debugging)
    :return: the deglycosylated molecule or a graph of rings
    """

    if len(Chem.GetMolFrags(mol)) > 1:
        raise ValueError(
            "Error! Deglycosylation is designed to work on single molecules, not mixtures!"
        )

    if mode not in ("run", "graph"):
        raise AttributeError(
            f"Error! Unauthorized value for parameter 'mode'! ('{mode}')"
        )

    # avoid inplace modifications
    mol = Chem.Mol(mol)

    # define rings
    rings = mol.GetRingInfo().AtomRings()
    rings = fuse_rings(rings)
    # try to deglycosylate only if the molecule has at least 2 rings:
    # - leave linear compounds out
    # - leave sugars in case they are the only ring on the molecule
    if len(rings) < 2:
        return mol

    # annotate sugar-like rings
    are_sugar_like = [_is_sugar_like(x, mol) for x in rings]
    # remove sugars only when the molecule has some sugar rings and is not entirely composed of sugars
    if not any(are_sugar_like) or all(are_sugar_like):
        return mol
    ring_atoms = set([item for sublist in rings for item in sublist])

    # init sugar graph
    G = Graph()
    # init linkers parts from left and right the glycosidic bond oxygen: one of the side is required to have either C or nothing
    authorized_linker_parts = [
        [],
        ["C"],
    ]  # R1-OxxxxR2 or R1-COxxxxR2 with xxxx being any sequence of linear atoms (same for R2->R1)

    # define linker atoms as shortest path between 2 rings that do not include other rings
    for i in range(len(rings)):
        ring1 = rings[i]
        for j in range(i + 1, len(rings)):
            ring2 = rings[j]

            # shortest path between the two rings that do not include the current rings themselves
            shortest_path = [
                x
                for x in Chem.GetShortestPath(mol, ring1[0], ring2[0])
                if x not in ring1 + ring2
            ]
            # define the other ring atoms
            other_ring_atoms = ring_atoms.symmetric_difference(set(ring1 + ring2))
            # shortest path for going from the left (ring1) to the right (ring2)
            shortest_path_elements = [
                mol.GetAtomWithIdx(x).GetSymbol() for x in shortest_path
            ]

            # in case ring1 (left) or/and ring2 (right) is sugar-like, append the side chains left and right
            # to the oxygen to the corresponding ring atoms to avoid left-overs (the O remains is not removed)
            glycosidic_bond = False
            if (
                "O" in shortest_path_elements
            ):  # not expected to be common enough for a try/catch statement
                # from the left side
                aidx_oxygen_left = shortest_path_elements.index(
                    "O"
                )  # first O found in list
                if (
                    are_sugar_like[i]
                    and shortest_path_elements[:aidx_oxygen_left]
                    in authorized_linker_parts
                ):
                    glycosidic_bond = True
                    ring1 += shortest_path[:aidx_oxygen_left]

                # from the right side
                shortest_path_elements.reverse()
                shortest_path.reverse()
                aidx_oxygen_right = shortest_path_elements.index(
                    "O"
                )  # first O found in list
                if (
                    are_sugar_like[j]
                    and shortest_path_elements[:aidx_oxygen_right]
                    in authorized_linker_parts
                ):
                    glycosidic_bond = True
                    ring2 += shortest_path[:aidx_oxygen_right]

            # in case the 2 rings are directly connected, append a new edge to G
            if not set(shortest_path).intersection(other_ring_atoms):
                G.add_edge(
                    i,
                    j,
                    atoms="".join(shortest_path_elements),
                    glycosidic_bond=glycosidic_bond,
                )
                # annotate nodes with the ring atoms (+ relevent linker atoms) and if they are sugar-like
                G.nodes[i]["atoms"] = ring1
                G.nodes[i]["sugar_like"] = are_sugar_like[i]
                G.nodes[j]["atoms"] = ring2
                G.nodes[j]["sugar_like"] = are_sugar_like[j]

    # draw the graph
    if mode == "graph":
        # colormap_nodes = [(0.7,0.7,0.7) if x['sugar_like'] else (1,0,0) for i, x in G.nodes(data=True)]
        # return draw.fc_graph(G, colormap_nodes=colormap_nodes)
        return G

    # iterative recording of terminal sugar rings (atoms) that are linked with a glycosidic bond
    ring_atoms_to_remove = []
    nodes_to_remove = [
        node
        for node in G.nodes(data=True)
        if node[1]["sugar_like"]
        and G.degree(node[0]) == 1
        and list(G.edges(node[0], data=True))[0][2]["glycosidic_bond"]
    ]
    while len(nodes_to_remove) > 0:
        # record atoms indices to remove from the molecule
        [ring_atoms_to_remove.append(n[1]["atoms"]) for n in nodes_to_remove]
        # remove nodes from current layer for next iteration
        [G.remove_node(n[0]) for n in nodes_to_remove]
        nodes_to_remove = [
            node
            for node in G.nodes(data=True)
            if node[1]["sugar_like"]
            and G.degree(node[0]) == 1
            and list(G.edges(node[0], data=True))[0][2]["glycosidic_bond"]
        ]

    # edit the molecule
    if ring_atoms_to_remove:
        # flatten the atom indices of each ring to remove in reverse order so that atom indices do not change when removing atoms
        ring_atoms_to_remove = sorted(
            [item for sublist in ring_atoms_to_remove for item in sublist], reverse=True
        )
        emol = Chem.EditableMol(mol)
        [emol.RemoveAtom(x) for x in ring_atoms_to_remove]
        mol = emol.GetMol()

    # clean-up
    frags = Chem.GetMolFrags(mol, asMols=True)
    # avoid counting the number of rings in each fragment if only 1 fragment left anyway
    if len(frags) == 1:
        return frags[0]
    # the substituents of the deleted terminal sugar-like rings remain in the structure,
    # these are obligatory linear because they were not in the graph,
    # so we just have to retrieve the one fragment that is not linear
    return [x for x in frags if Descriptors.rdMolDescriptors.CalcNumRings(x) > 0][0]
