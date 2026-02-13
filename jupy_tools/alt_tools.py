# -*- coding: utf-8 -*-
"""
Tools and extensions for the Altair plotting library.
"""

import os
import os.path as op

import pandas as pd

from jupy_tools import mol_view as mv


def add_img_refs(df, id_col, image_dir="images"):
    """
    Adds a column with image references to the dataframe based on the Smiles column and an identifier column.
    The images are saved in the specified directory with filenames based on the identifier column.
    The column `image` can then be used in Altair tooltips to display the molecule images.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing at least the columns specified in `id_col` and `Smiles`.
    id_col : str
        The name of the column in the dataframe that contains unique identifiers for each molecule (e.g., 'ID').
    image_dir : str, optional
        The directory where the molecule images will be saved. Default is 'images'.

    Returns:
    --------
    pd.DataFrame
        The input dataframe with an additional column 'image' containing the file paths to the molecule images.
    """
    img_refs = []
    ids = []
    os.makedirs(image_dir, exist_ok=True)
    for _, row in df.iterrows():
        img_ref = f"{image_dir}/{row[id_col]}.png"
        fn = img_ref
        if op.exists(fn):
            img_refs.append(img_ref)
            ids.append(row[id_col])
            continue
        img = mv.MolImage(row["Smiles"], svg=False).txt
        open(fn, "wb").write(img)
        img_refs.append(img_ref)
        ids.append(row[id_col])
    img_tmp = pd.DataFrame({id_col: ids, "image": img_refs})
    df = pd.merge(df, img_tmp, on=id_col, how="left")
    return df
