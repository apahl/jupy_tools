#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My favourite Matplotlib styles."""

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
# plt.style.use("seaborn-v0_8-white")
# plt.style.use("seaborn-pastel")
plt.style.use("seaborn-v0_8-poster")  # seaborn-talk
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.title_fontsize"] = 20
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["svg.fonttype"] = "none"


def apply(name: str):
    """Apply the style to the current Matplotlib session."""
    assert name in [
        "default",
        "medium_dark",
        "medium_light",
        "small_dark",
        "small_light",
    ], f"Style '{name}' is not defined."
    if name == "default":
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.style.use("seaborn-v0_8-poster")  # seaborn-talk
        plt.rcParams["axes.titlesize"] = 25
        plt.rcParams["axes.labelsize"] = 25
        plt.rcParams["xtick.labelsize"] = 20
        plt.rcParams["ytick.labelsize"] = 20
        plt.rcParams["legend.title_fontsize"] = 20
        plt.rcParams["legend.fontsize"] = 15
        plt.rcParams["svg.fonttype"] = "none"
        return
    if "dark" in name:
        plt.style.use("dark_background")
    if "light" in name:
        plt.style.use("seaborn-v0_8-whitegrid")
    if "medium" in name:
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.title_fontsize"] = 12
        plt.rcParams["legend.fontsize"] = 10
        plt.rcParams["svg.fonttype"] = "none"
    if "small" in name:
        plt.rcParams["axes.titlesize"] = 10
        plt.rcParams["axes.labelsize"] = 10
        plt.rcParams["xtick.labelsize"] = 8
        plt.rcParams["ytick.labelsize"] = 8
        plt.rcParams["legend.title_fontsize"] = 10
        plt.rcParams["legend.fontsize"] = 8
        plt.rcParams["svg.fonttype"] = "none"
