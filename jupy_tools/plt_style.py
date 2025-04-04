#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My favourite Matplotlib styles."""

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
# plt.style.use("seaborn-white")
# plt.style.use("seaborn-pastel")
plt.style.use("seaborn-v0_8-poster")  # seaborn-talk
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.title_fontsize"] = 20
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["svg.fonttype"] = "none"
