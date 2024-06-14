#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for viewing DataFrames with molecules.
"""
# import os.path as op

import base64
import time
from itertools import chain
from io import BytesIO as IO
import os
import os.path as op
from typing import List, Optional, Union

import pandas as pd
import numpy as np

from PIL import Image, ImageChops

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdCoordGen import AddCoords  # New coord. generation
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from . import utils, templ

USE_RDKIT_NEW_COORD = True
BGCOLOR = "#94CAEF"
IMG_GRID_SIZE = 235
SVG = True
THEME = "light"  # "light" or "dark" (for SVG structure display)

pd.set_option("display.max_colwidth", None)


def is_interactive_ipython():
    try:
        get_ipython()  # type: ignore
        ipy = True
        # print("> interactive IPython session.")
    except NameError:
        ipy = False
    return ipy


IPYTHON = is_interactive_ipython()
if IPYTHON:
    from IPython.core.display import HTML

# Enable some additional functionality if the matplotlib and seaborn are available:
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import gc

    CHARTS = True
except ImportError:
    CHARTS = False


def rescale(mol, f=1.4):
    tm = np.zeros((4, 4), np.double)
    for i in range(3):
        tm[i, i] = f
    tm[3, 3] = 1.0
    Chem.TransformMol(mol, tm)


def add_coords(mol, force=False):
    """Check if a mol has 2D coordinates and if not, calculate them."""
    if not force:
        try:
            mol.GetConformer()
        except ValueError:
            force = True  # no 2D coords... calculate them

    if force:
        if USE_RDKIT_NEW_COORD and mol.GetNumAtoms() <= 75:
            AddCoords(mol)
            rescale(mol, f=1.4)
        else:
            mol.Compute2DCoords()


def make_transparent(img):
    img = img.convert("RGBA")
    pixdata = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)
    return img


def autocrop(im, bgcolor="white"):
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg = Image.new("RGB", im.size, bgcolor)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return None  # no contents


def b64_mol(img_file):
    b64 = base64.b64encode(img_file)
    b64 = b64.decode()
    return b64


def b64_fig(chart, format="PNG"):
    fig = chart.get_figure()
    img_file = IO()
    fig.savefig(img_file, format=format, bbox_inches="tight")
    b64 = base64.b64encode(img_file.getvalue())
    b64 = b64.decode()
    img_file.close()
    return b64


class DrawingOptions:
    def __init__(self):
        """Create a default DrawingOptions instance."""
        self.size = 300
        self.use_colors = True
        self.add_atom_indices = False
        self.add_bond_indices = False
        self.add_stereo_annotation = False
        self.annotation_font_scale = 0.75  # RDKit default: 0.5
        self.base_font_size = -1.0
        self.bond_line_width = 2.0
        self.max_font_size = 40
        self.min_font_size = 6
        self.multiple_bond_offset = 0.20  # RDKit default: 0.15

    def update(self, dos):
        """Update the given RDKit DrawingOptions instance `dos` with the options set in this instance.
        The size is not updated, because it has to be set when the RDKit Canvas instance (`MolDraw2DXXX`) is created.
        To emphasize: the RDKit instance `dos` is updated, not this instance."""
        if not self.use_colors:
            dos.useBWAtomPalette()
        dos.addAtomIndices = self.add_atom_indices
        dos.addBondIndices = self.add_bond_indices
        dos.addStereoAnnotation = self.add_stereo_annotation
        dos.annotationFontScale = self.annotation_font_scale
        dos.baseFontSize = self.base_font_size
        dos.bondLineWidth = self.bond_line_width
        dos.maxFontSize = self.max_font_size
        dos.minFontSize = self.min_font_size
        dos.multipleBondOffset = self.multiple_bond_offset

    def __repr__(self):
        attr = ["DrawingOtions:"]
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            attr.append(f"{k}={v}")
        return "\n".join(attr)


class MolImage:
    def __init__(
        self,
        mol: Union[Mol, str],
        svg: Optional[bool] = None,
        hlsss: Optional[str] = None,
        options: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate an image (SVG or PNG) of a molecule.
        After calling the constructor, the image is stored in the `txt` attribute.
        A HTML image tag is available in the `tag` attribute.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol or Smiles string
            Molecule to draw.
        size : int
            Size of the image in pixels (default=300).
        svg : bool or None
            If True, the image is saved as SVG. Otherwise, it is saved as PNG.
        hlsss : str or None
            Highlight the substructure given as Smiles.
        options : str or None
            Additional HTML options for the drawing.
        KWargs:
            alt_text: whether to include Smiles (default) or the MolBlock as alt text.
                Use e.g. the firefox extension `Copy Image Text` to copy the alt text.

        Image format options:
            There are two ways to format the structure images:
            1. Use the `size` and `use_colors` "quick access" options.
            2. Pass a DrawingOptions dictionary as `drawing_options`. Use the function init_draw_options() to create a dict with default values.
            When the second option is used, the `size` and `use_colors` options are ignored.
            Each of the options can be either a single value or a list of values. If a list is passed,
            it has to have the same length as the number of Smiles columns.

            size: Size of the image in pixels (default=300).
            use_colors: Whether to use colors for atoms (default=True).
            drawing_options: instance of `DrawingOptions` for the structure display.
        """
        size = kwargs.get("size", 300)
        use_colors = kwargs.get("use_colors", True)
        drawing_options = kwargs.get("drawing_options", None)

        # print(f"{size=}")
        # print(f"{use_colors=}")
        # print(f"{drawing_options=}")
        # print("------------------------------")
        self.mol = mol
        self.smiles = None  # used for the alternative text in the img tag
        if isinstance(self.mol, str):  # convert from Smiles on-the-fly, when necessary
            if len(self.mol) > 0:
                self.smiles = mol
                self.mol = Chem.MolFromSmiles(mol)
            else:
                self.smiles = "*"
                self.mol = Chem.MolFromSmiles("*")
        else:
            self.smiles = Chem.MolToSmiles(mol)
        if self.mol is None or self.mol is np.nan:
            self.smiles = "*"
            self.mol = Chem.MolFromSmiles("*")

        if drawing_options is not None:
            self.size = drawing_options.size
        else:
            self.size = size
        self.svg = svg if svg is not None else SVG
        assert isinstance(self.svg, bool)
        self.hlsss = hlsss
        alt_text = kwargs.get("alt_text", "smiles").lower()

        if hlsss is not None:
            if isinstance(hlsss, str):
                hlsss = hlsss.split(",")
                atoms = set()
                for smi in hlsss:
                    m = Chem.MolFromSmiles(smi)
                    if m:
                        matches = list(chain(*self.mol.GetSubstructMatches(m)))
                    else:
                        matches = []
                    if len(matches) > 0:
                        atoms = atoms.union(set(matches))
            hl_atoms = {x: "#ff0000" for x in list(atoms)}
        else:
            hl_atoms = {}

        add_coords(self.mol)
        if "smiles" in alt_text:
            self.alt_text = self.smiles
        else:
            self.alt_text = Chem.MolToMolBlock(self.mol)
        if self.svg:
            d2d = rdMolDraw2D.MolDraw2DSVG(size, size)
        else:
            d2d = rdMolDraw2D.MolDraw2DCairo(size, size)
        dos = d2d.drawOptions()
        if drawing_options is None:
            if not use_colors:
                dos.useBWAtomPalette()
            dos.multipleBondOffset = 0.20
        else:
            drawing_options.update(dos)
        if hlsss is None:
            d2d.DrawMolecule(self.mol)
        else:
            d2d.DrawMoleculeWithHighlights(self.mol, "", hl_atoms, {}, {}, {})
        d2d.FinishDrawing()
        img = d2d.GetDrawingText()
        if self.svg:
            # remove the opaque background ("<rect...") and skip the first line with the "<xml>" tag ("[1:]")
            img_list = [
                line for line in img.splitlines()[1:] if not line.startswith("<rect")
            ]
            img = "\n".join(img_list)
            if THEME == "dark":
                img = img.replace("#000000", "#FFFFFF")

        else:
            try:
                img = Image.open(IO(img))
                img = autocrop(img)
            except UnicodeEncodeError:
                print(Chem.MolToSmiles(mol))
                mol = Chem.MolFromSmiles("*")
                img = autocrop(Draw.MolToImage(mol, size=(size, size)))
            img = make_transparent(img)
            img_file = IO()
            img.save(img_file, format="PNG")
            val = img_file.getvalue()
            img_file.close()
            img = val

        self.txt = img

        if options is None:
            options = ""
        if self.svg:
            img = bytes(self.txt, encoding="iso-8859-1")
            img = b64_mol(img)
            tag = """<img {} src="data:image/svg+xml;base64,{}" alt="{}"/>"""
            self.tag = tag.format(options, img, self.alt_text)
        else:
            tag = """<img {} src="data:image/png;base64,{}" alt="{}"/>"""
            self.tag = tag.format(options, b64_mol(img), self.alt_text)

    def save(self, fn):
        if (self.svg and not fn.lower().endswith("svg")) or (
            not self.svg and not fn.lower().endswith("png")
        ):
            img_fmt = "SVG" if self.svg else "PNG"
            raise ValueError(
                f"file ending of {fn} does not match drawing format {img_fmt}."
            )
        mode = "w" if self.svg else "wb"
        with open(fn, mode) as f:
            f.write(self.txt)


def write(text, fn):
    with open(fn, "w") as f:
        f.write(text)


# def _mol_img_tag(mol):
#     return pd.Series(mol_img_tag(mol))


def add_image_tag(
    df,
    col,
    smiles_col="Smiles",
    svg=None,
    options=None,
    **kwargs,
):
    """
    Add an image tag to a dataframe column.
    The image tag can be used to display the molecule in a web page.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the molecules.
    col : str
        Name of the column containing the images.
    size : int
        Size of the image in pixels (default=300).
    svg : bool or None
        If True, the image is saved as SVG. Otherwise, it is saved as PNG.
        If None, the module default is used.
    options : str or None
        Additional HTML options for the drawing.
    kwargs: are passed to the MolImage constructor.
    """
    df = utils.calc_from_smiles(
        df,
        col,
        lambda x: MolImage(
            x,
            svg=svg,
            options=options,
            **kwargs,
        ).tag,
        smiles_col=smiles_col,
        filter_nans=False,
    )
    return df


def _apply_link(input, link, ln_title="Link"):
    """input[0]: mol_img_tag
    input[1]: link_col value"""
    link_str = link.format(input[1])
    result = '<a target="_blank" href="{}" title="{}">{}</a>'.format(
        link_str, ln_title, input[0]
    )
    return result


def show_mols(mols_or_smiles, cols=5, size=IMG_GRID_SIZE, svg=None):
    """A small utility to quickly view a list of mols (or Smiles) in a grid."""
    if svg is None:
        svg = SVG
    assert isinstance(svg, bool)
    html_list = ['<table align="center">']
    idx = 0
    row_line = []
    if not isinstance(mols_or_smiles, list):
        mols_or_smiles = [mols_or_smiles]
    for mol in mols_or_smiles:
        idx += 1
        mol_img = MolImage(mol, size=size, svg=svg)
        cell = "<td>{}<td>".format(mol_img.tag)
        row_line.append(cell)
        if idx == cols:
            row = "<tr>" + "".join(row_line) + "</td>"
            html_list.append(row)
            idx = 0
            row_line = []
    if idx != 0:
        row = "<tr>" + "".join(row_line) + "</td>"
        html_list.append(row)
    html_list.append("</table>")
    table = "\n".join(html_list)
    if IPYTHON:
        return HTML(table)
    return table


def mol_grid(
    df,
    drop=[],
    keep=[],
    hide=[],
    smiles_col="Smiles",
    mol_col="Mol",
    id_col="Compound_Id",
    as_html=True,
    bar: Optional[list[str]] = None,
    svg=None,
    img_folder=None,
    **kwargs,
):
    """Creates a HTML grid out of the DataFrame input.

    Parameters:
        df (DataFrame): Pandas DataFrame with either a Smiles column or a Mol column.
        interactive (bool)
        as_html: return a Jupyter HTML object, when possible, otherwise return a string.
        link_templ, link_col (str) (then interactive is false)
        bar [Option[list[str]]: displays the listed columns as bar chart in the grid.
            Y-limits can be set with the `ylim` tuple.

    Image format options:
        There are two ways to format the structure images:
        1. Use the `size` and `use_colors` "quick access" options.
        2. Pass a DrawingOptions dictionary as `drawing_options`. Use the function init_draw_options() to create a dict with default values.
        When the second option is used, the `size` and `use_colors` options are ignored.
        Each of the options can be either a single value or a list of values. If a list is passed,
        it has to have the same length as the number of Smiles columns.

        size: Size of the image in pixels (default=300).
        use_colors: Whether to use colors for atoms (default=True).
        drawing_options: instance of `DrawingOptions` for the structure display.

        The image options are passed directly to the MolImage constructor.

    Returns:
        HTML table as TEXT with molecules in grid-like layout to embed in IPython or a web page.
    """

    if svg is None:
        svg = SVG
    assert isinstance(svg, bool)

    if bar is not None:
        # Cluster profiles can only be shown when the Cell Painting module is available.
        if not CHARTS:
            bar = None
    if bar is not None:
        assert isinstance(bar, list)
        hide.extend(bar)
        bar_names_max_len = max([len(x) for x in bar])

    if img_folder is not None:
        os.makedirs(img_folder, exist_ok=True)

    interact = kwargs.pop("interactive", False)
    link_templ = kwargs.pop("link_templ", None)
    link_col = kwargs.pop("link_col", None)
    if link_col is not None:
        interact = (
            False  # interact is not avail. when clicking the image should open a link
        )
    mols_per_row = kwargs.pop("mols_per_row", 5)
    hlsss = kwargs.pop(
        "hlsss", None
    )  # colname with Smiles (,-separated) for Atom highlighting
    truncate = kwargs.pop("truncate", 25)
    ylim = kwargs.pop("ylim", None)
    if "size" not in kwargs:
        kwargs["size"] = IMG_GRID_SIZE
    size = kwargs["size"]

    df = df.copy()
    if mol_col not in df.keys():
        df = utils.add_mol_col(df, smiles_col=smiles_col)
    if len(keep) > 0:
        keep.append(mol_col)
        if id_col is not None and id_col not in keep:
            keep.append(id_col)
        df = df[keep]
    drop.append(smiles_col)
    df = utils.drop_cols(df, drop)
    props = []
    for x in list(df.keys()):
        if x != mol_col and x != id_col and x != hlsss and (x not in hide):
            props.append(x)
    time_stamp = time.strftime("%y%m%d%H%M%S")
    td_opt = {"style": "text-align: center;"}
    header_opt = {"bgcolor": BGCOLOR}
    table_list = []
    guessed_id = id_col
    if guessed_id not in df.keys():
        guessed_id = None
    if interact and guessed_id is not None:
        table_list.append(templ.TBL_JAVASCRIPT.format(ts=time_stamp, bgcolor=BGCOLOR))

    if len(props) > 0:
        td_opt["colspan"] = "2"
        prop_row_cells = {k: [] for k, _ in enumerate(props)}

    rows = []
    id_cells = []
    mol_cells = []
    chart_cells = []
    for idx, (_, rec) in enumerate(df.iterrows(), 1):
        mol = rec[mol_col]
        if guessed_id:
            id_prop_val = str(rec[guessed_id])
            img_id = id_prop_val
            cell_opt = {"id": "{}_{}".format(id_prop_val, time_stamp)}
            cell_opt.update(td_opt)
            cell_opt.update(header_opt)
            id_cells.extend(templ.td(id_prop_val, cell_opt))
        else:
            img_id = idx

        if mol is None or mol is np.nan:
            cell = ["no structure"]

        else:
            if hlsss is not None:
                hlsss_smi = rec[hlsss]
            else:
                hlsss_smi = None

            if interact and guessed_id is not None:
                img_opt = {
                    "title": "Click to select / unselect",
                    "onclick": "toggleCpd('{}')".format(id_prop_val),
                }
            elif link_col is not None:
                img_opt = {"title": "Click to open link"}
                #          "onclick": "location.href='{}';".format(link)}
                # '<a target="_blank" href="{}" title="{}">{}</a>'

            else:
                img_opt = {"title": str(img_id)}

            img_opt["style"] = f"max-width: {size}px; max-height: {size}px;"

            # img_opt["height"] = "{}px".format(size)
            # cell = templ.img(img_src, img_opt)
            if svg:
                img_ext = "svg"
                img_size = size
            else:
                img_ext = "png"
                img_size = size * 2
            kwargs["size"] = img_size
            if img_folder is not None and guessed_id is not None:
                img_fn = op.join(img_folder, f"{rec[guessed_id]}.{img_ext}")
            else:
                img_fn = None
            img_opt = " ".join([f'{k}="{str(v)}"' for k, v in img_opt.items()])
            mol_img = MolImage(mol, svg=svg, hlsss=hlsss_smi, options=img_opt, **kwargs)
            if img_fn is not None:
                mol_img.save(img_fn)
            cell = mol_img.tag

            if link_col is not None:
                if isinstance(link_col, str):
                    fields = [link_col]
                else:
                    fields = [rec[x] for x in link_col]
                link = link_templ.format(*fields)
                a_opt = {"href": link}
                cell = templ.a(cell, a_opt)

        # td_opt = {"align": "center"}
        td_opt = {
            "style": "text-align: center;",
            "bgcolor": "#FFFFFF",
        }
        if len(props) > 0:
            td_opt["colspan"] = "2"

        mol_cells.extend(templ.td(cell, td_opt))

        if bar is not None:
            plt.figure(figsize=(6, 6))
            bar_values = rec[bar].values
            chart = sns.barplot(x=bar, y=bar_values, color="#94caef")
            if bar_names_max_len > 1:
                plt.xticks(rotation=90)
            if ylim is not None:
                plt.ylim(ylim)
            bf = b64_fig(chart)
            img_width = IMG_GRID_SIZE + int(IMG_GRID_SIZE * 0.2)
            chart_tag = f"""<img width="{img_width}" src="data:image/png;base64,{bf}" alt="Chart"/>"""
            chart_cells.extend(templ.td(chart_tag, td_opt))

            plt.clf()
            plt.close()
            gc.collect()

        if len(props) > 0:
            for prop_no, prop in enumerate(props):
                prop_opt = {"style": "text-align: left;"}
                val_opt = {"style": "text-align: left;"}
                prop_cells = []
                prop_val = ""
                if prop in rec:
                    prop_val = str(rec[prop])
                    if (
                        prop == "Pure_Flag"
                        and prop_val != ""
                        and prop_val != "n.d."
                        and "Purity" in rec
                        and "LCMS_Date" in rec
                    ):
                        val_opt["title"] = "{}% ({})".format(
                            rec["Purity"], rec["LCMS_Date"]
                        )
                prop_cells.extend(templ.td(prop[:25], prop_opt))
                prop_cells.extend(
                    templ.td(
                        templ.div(
                            prop_val[:truncate],
                            options={"style": f"max-width: {size-20}px;"},
                        ),
                        val_opt,
                    ),
                )
                prop_row_cells[prop_no].extend(prop_cells)

        if idx % mols_per_row == 0 or idx == len(df):
            if guessed_id:
                rows.extend(templ.tr(id_cells))
            rows.extend(templ.tr(mol_cells))
            if bar is not None:
                rows.extend(templ.tr(chart_cells))

            if len(props) > 0:
                colspan_factor = 2
                for prop_no in sorted(prop_row_cells):
                    rows.extend(templ.tr(prop_row_cells[prop_no]))
                prop_row_cells = {k: [] for k, _ in enumerate(props)}
            else:
                colspan_factor = 1
            empty_row_options = {"colspan": mols_per_row * colspan_factor}
            empty_row_options["style"] = "border: none;"
            empty_row = templ.tr(templ.td("&nbsp;", options=empty_row_options))
            rows.extend(empty_row)
            id_cells = []
            mol_cells = []
            chart_cells = []

    table_list.extend(templ.table(rows))

    if interact and guessed_id is not None:
        table_list.append(templ.ID_LIST.format(ts=time_stamp))

    # print(table_list)
    if IPYTHON and as_html:
        return HTML("".join(table_list))
    return "".join(table_list)


def write_mol_grid(
    df,
    title="MolGrid",
    fn="molgrid.html",
    drop=[],
    keep=[],
    smiles_col="Smiles",
    mol_col="Mol",
    id_col="Compound_Id",
    write_images=False,
    svg=None,
    **kwargs,
):
    """
    Write a grid of molecules to a file and return the link.

    Args:
        df: DataFrame with molecules.
        title: Document title.
        fn: Filename to write.

    KWargs:
        alt_text: whether to include Smiles (default) or the MolBlock as alt text.
            Use e.g. the firefox extension `Copy Alt Text` to copy the alt text.

    Image format options:
        There are two ways to format the structure images:
        1. Use the `size` and `use_colors` "quick access" options.
        2. Pass a DrawingOptions dictionary as `drawing_options`. Use the function init_draw_options() to create a dict with default values.
        When the second option is used, the `size` and `use_colors` options are ignored.
        Each of the options can be either a single value or a list of values. If a list is passed,
        it has to have the same length as the number of Smiles columns.

        size: Size of the image in pixels (default=300).
        use_colors: Whether to use colors for atoms (default=True).
        drawing_options: instance of `DrawingOptions` for the structure display.
    """

    if svg is None:
        svg = SVG
    assert isinstance(svg, bool)

    if "size" not in kwargs:
        kwargs["size"] = 300

    # Need to check here, because the `add_image_tag` function will run in a try / except block:
    alt_text = kwargs.get("alt_text", "smiles").lower()
    if "smiles" not in alt_text and "block" not in alt_text:
        raise ValueError(
            f"Unknown value for `alt_text`: {alt_text}. Must be 'smiles' (default) or 'molblock'."
        )

    if write_images:
        html_dir = op.dirname(fn)
        img_folder = op.join(html_dir, "images")
    else:
        img_folder = None

    tbl = mol_grid(
        df,
        title=title,
        drop=drop,
        keep=keep,
        smiles_col=smiles_col,
        mol_col=mol_col,
        id_col=id_col,
        as_html=False,
        img_folder=img_folder,
        svg=svg,
        **kwargs,
    )
    header = kwargs.pop("header", None)
    summary = kwargs.pop("summary", None)

    page = templ.page(tbl, title=title, header=header, summary=summary)
    utils.write(page, fn)
    if IPYTHON:
        return HTML('<a href="{}">{}</a>'.format(fn, title))


def write_mol_table(
    df: pd.DataFrame,
    title: str = "MolTable",
    fn: str = "moltable.html",
    id_col="Compound_Id",
    smiles_col: str | List[str] = "Smiles",
    size: int | List[int] = 300,
    use_colors: bool | List[bool] = True,
    drawing_options: Optional[DrawingOptions | List[DrawingOptions]] = None,
    svg: Optional[bool] = None,
    formatter=None,
    **kwargs,
):
    """
    Write a table of molecules to a file and return the link.

    Args:
        df: DataFrame with molecules.
        title: Document title.
        fn: Filename to write.
        smiles_col: Column name or list of column names with Smiles.
            Each column will be displayed as a separate molecule.

    KWargs:
        alt_text: whether to include Smiles (default) or the MolBlock as alt text.
            Use e.g. the firefox extension `Copy Alt Text` to copy the alt text.

    Image format options:
        There are two ways to format the structure images:
        1. Use the `size` and `use_colors` "quick access" options.
        2. Pass a DrawingOptions dictionary as `drawing_options`. Use the function init_draw_options() to create a dict with default values.
        When the second option is used, the `size` and `use_colors` options are ignored.
        Each of the options can be either a single value or a list of values. If a list is passed,
        it has to have the same length as the number of Smiles columns.

        size: Size of the image in pixels (default=300). If this is a list,
            each image column will be displayed with the corresponding size.
        use_colors: Whether to use colors for atoms (default=True). If this is a list,
            each image column will be displayed with the corresponding setting.
        drawing_options: instance of `DrawingOptions` for the structure display.
            If this is a list, each image column will be displayed with the corresponding options.
    """
    df = df.copy()
    header = kwargs.pop("header", None)
    summary = kwargs.pop("summary", None)

    # Need to check here, because the `add_image_tag` function will run in a try / except block:
    alt_text = kwargs.get("alt_text", "smiles").lower()
    if "smiles" not in alt_text and "block" not in alt_text:
        raise ValueError(
            f"Unknown value for `alt_text`: {alt_text}. Must be 'smiles' (default) or 'molblock'."
        )

    # Some sanity checks:
    if svg is None:
        svg = SVG
    assert isinstance(svg, bool)
    if isinstance(smiles_col, str):
        smiles_col = [smiles_col]
    for sc in smiles_col:
        assert sc in df.keys(), f"Column {sc} not found in DataFrame."
    if isinstance(size, int):
        size = [size] * len(smiles_col)
    if isinstance(use_colors, bool):
        use_colors = [use_colors] * len(smiles_col)
    if not isinstance(drawing_options, list):
        drawing_options = [drawing_options] * len(smiles_col)
    assert len(size) == len(smiles_col)
    assert id_col in df.keys(), f"Id Column {id_col} not found in DataFrame."

    cols = df.keys()
    # When there is only one structure column, put it in front:
    if len(smiles_col) == 1:
        cols = [smiles_col[0]] + [x for x in cols if x != smiles_col[0]]

    # Add the image tags
    for idx, sc in enumerate(smiles_col):
        df = add_image_tag(
            df,
            f"{sc}_Mol",
            size=size[idx],
            use_colors=use_colors[idx],
            drawing_options=drawing_options[idx],
            svg=svg,
            smiles_col=sc,
            **kwargs,
        )

    # Replace the positions of the Smiles columns with the Mol columns:
    for idx, sc in enumerate(smiles_col):
        cols = [f"{sc}_Mol" if x == sc else x for x in cols]

    # Drop the Smiles columns:
    df = df[cols]
    if formatter is None:
        # Create a default formatter that displays floats with 3 decimals:
        float_cols = [x for x in df.keys() if df[x].dtype == "float64"]
        formatter = {x: "{:.3f}" for x in float_cols}
    style = (
        df.style.format(formatter=formatter, escape=None)
        .set_table_styles([templ.TABLE, templ.HEADERS])
        .set_properties(**templ.CELLS)
    )

    # style.to_html("output/stats_fcc/fcc_diff_aminergic_gpcr.html", escape=False)
    html = style.to_html(None)
    lines = []
    for line in html.split("\n"):
        if "thead" in line:
            continue
        if "tbody" in line:
            continue
        lines.append(line)
    html = "\n".join(lines)
    html = html + templ.TABLE_SORTER
    page = templ.page(html, title=title, header=header, summary=summary)
    utils.write(page, fn)
    if IPYTHON:
        return HTML('<a href="{}">{}</a>'.format(fn, title))
