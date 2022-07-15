# -*- coding: utf-8 -*-
"""
Extensions for holoviews and hvplot.
"""

from . import utils as u, mol_view as mv

import holoviews as hv

hv.extension("bokeh")
from bokeh.models import HoverTool


def struct_hover(df, force=False, cols=[]):
    """Create a structure tooltip that can be used in Holoviews.
    Takes a MolFrame instance as parameter."""
    image_hover = ""
    if "Smiles" in df.columns:
        if force or "Image" not in df.columns:
            df = u.calc_from_smiles(
                df, "Image", lambda x: mv.MolImage(x, options='width="75%"').tag
            )
        image_hover = """<div>
                    @Image<br>
                </div>"""
    add_cols = []
    for col in cols:
        if col in df.columns:
            add_cols.append(
                f"""<div>
                        <span style="font-size: 12px;">{col}: @{col}</span>
                    </div>"""
            )
    add_cols_txt = "\n".join(add_cols)
    hover = HoverTool(
        tooltips=f"""
            <div>
                {image_hover}
                {add_cols_txt}
            </div>
        """
    )
    return df, hover


def scatter(
    df,
    x,
    y,
    colorby=None,
    options={},
    styles={},
    tooltip=None,
    title="Scatter Plot",
    force=False,
):
    """Possible options: width, height, legend_position [e.g. "top_right"]
    Possible styles: size, cmap [brg, Accent, rainbow, jet, flag, Wistia]
    Only works when the data object is a Pandas DataFrame."""
    df = df.copy()
    if tooltip is None:
        tooltip = []
    if isinstance(tooltip, str):
        tooltip = [tooltip]
    if x not in tooltip:
        tooltip.append(x)
    if y not in tooltip:
        tooltip.append(y)
    if colorby is not None and colorby not in tooltip:
        tooltip.append(colorby)
    df, hover = struct_hover(df, force=force, cols=tooltip)
    plot_options = {"width": 800, "height": 600, "tools": [hover]}
    plot_styles = {"size": 8, "cmap": "brg"}
    plot_options.update(options)
    plot_styles.update(styles)
    kdims = [x]
    vdims = [y]
    if "Smiles" in df.columns:
        vdims.append("Image")
    for tt in tooltip:
        if tt not in vdims:
            vdims.append(tt)
    if colorby is None:
        scatter = hv.Scatter(data=df, kdims=kdims, vdims=vdims, label=title)
    else:
        if colorby not in vdims:
            vdims.append(colorby)
        df = df.sort_values(colorby)
        df[colorby] = df[colorby].astype(str)
        scatter = hv.Scatter(data=df, kdims=kdims, vdims=vdims, label=title)
    plot_options["legend_position"] = "right"
    plot_options["toolbar"] = "above"
    plot_styles["color"] = colorby
    opts = {"Scatter": {"plot": plot_options, "style": plot_styles}}
    return scatter.opts(opts)


def save(plot, filename):
    """Save a plot to a file.
    This is just a convenience function, so that the calling code
    does not have to import holoviews."""
    hv.save(plot, filename)
