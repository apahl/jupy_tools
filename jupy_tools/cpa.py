#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for analysis of the Cell Painting Assay data.
"""

import functools
from glob import glob
import os.path as op
from typing import Iterable, List, Optional, Union

# import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import scipy.spatial.distance as dist

OUTPUT_DIR = "../output"

ACT_PROF_FEATURES = [
    "Median_Cells_AreaShape_Area",
    "Median_Cells_AreaShape_MaximumRadius",
    "Median_Cells_AreaShape_MeanRadius",
    "Median_Cells_AreaShape_MinFeretDiameter",
    "Median_Cells_AreaShape_MinorAxisLength",
    "Median_Cells_AreaShape_Perimeter",
    "Median_Cells_Correlation_Correlation_ER_Ph_golgi",
    "Median_Cells_Correlation_Correlation_ER_Syto",
    "Median_Cells_Correlation_Correlation_Hoechst_ER",
    "Median_Cells_Correlation_Correlation_Hoechst_Mito",
    "Median_Cells_Correlation_Correlation_Hoechst_Ph_golgi",
    "Median_Cells_Correlation_Correlation_Hoechst_Syto",
    "Median_Cells_Correlation_Correlation_Mito_ER",
    "Median_Cells_Correlation_Correlation_Mito_Ph_golgi",
    "Median_Cells_Correlation_Correlation_Mito_Syto",
    "Median_Cells_Correlation_Correlation_Syto_Ph_golgi",
    "Median_Cells_Correlation_K_ER_Syto",
    "Median_Cells_Correlation_K_Hoechst_Syto",
    "Median_Cells_Correlation_K_Mito_Hoechst",
    "Median_Cells_Correlation_K_Ph_golgi_Syto",
    "Median_Cells_Correlation_K_Syto_ER",
    "Median_Cells_Correlation_K_Syto_Hoechst",
    "Median_Cells_Correlation_K_Syto_Ph_golgi",
    "Median_Cells_Correlation_Manders_ER_Hoechst",
    "Median_Cells_Correlation_Manders_ER_Syto",
    "Median_Cells_Correlation_Manders_Mito_Hoechst",
    "Median_Cells_Correlation_Manders_Ph_golgi_Hoechst",
    "Median_Cells_Correlation_Manders_Syto_Hoechst",
    "Median_Cells_Correlation_Overlap_Hoechst_ER",
    "Median_Cells_Correlation_Overlap_Hoechst_Mito",
    "Median_Cells_Correlation_Overlap_Hoechst_Ph_golgi",
    "Median_Cells_Correlation_Overlap_Hoechst_Syto",
    "Median_Cells_Correlation_Overlap_Mito_ER",
    "Median_Cells_Correlation_Overlap_Mito_Ph_golgi",
    "Median_Cells_Correlation_Overlap_Mito_Syto",
    "Median_Cells_Correlation_Overlap_Syto_Ph_golgi",
    "Median_Cells_Correlation_RWC_ER_Mito",
    "Median_Cells_Correlation_RWC_Hoechst_ER",
    "Median_Cells_Correlation_RWC_Hoechst_Mito",
    "Median_Cells_Correlation_RWC_Hoechst_Ph_golgi",
    "Median_Cells_Correlation_RWC_Hoechst_Syto",
    "Median_Cells_Correlation_RWC_Mito_Hoechst",
    "Median_Cells_Correlation_RWC_Mito_Syto",
    "Median_Cells_Correlation_RWC_Ph_golgi_Hoechst",
    "Median_Cells_Correlation_RWC_Ph_golgi_Mito",
    "Median_Cells_Correlation_RWC_Ph_golgi_Syto",
    "Median_Cells_Correlation_RWC_Syto_Hoechst",
    "Median_Cells_Correlation_RWC_Syto_Mito",
    "Median_Cells_Granularity_1_Mito",
    "Median_Cells_Granularity_1_Ph_golgi",
    "Median_Cells_Granularity_1_Syto",
    "Median_Cells_Granularity_2_Mito",
    "Median_Cells_Granularity_2_Ph_golgi",
    "Median_Cells_Granularity_2_Syto",
    "Median_Cells_Granularity_3_ER",
    "Median_Cells_Granularity_3_Mito",
    "Median_Cells_Granularity_3_Ph_golgi",
    "Median_Cells_Granularity_3_Syto",
    "Median_Cells_Granularity_4_Mito",
    "Median_Cells_Granularity_5_Mito",
    "Median_Cells_Intensity_IntegratedIntensityEdge_Hoechst",
    "Median_Cells_Intensity_IntegratedIntensity_Syto",
    "Median_Cells_Intensity_LowerQuartileIntensity_Mito",
    "Median_Cells_Intensity_MADIntensity_ER",
    "Median_Cells_Intensity_MADIntensity_Hoechst",
    "Median_Cells_Intensity_MADIntensity_Mito",
    "Median_Cells_Intensity_MADIntensity_Ph_golgi",
    "Median_Cells_Intensity_MADIntensity_Syto",
    "Median_Cells_Intensity_MaxIntensityEdge_Mito",
    "Median_Cells_Intensity_MaxIntensityEdge_Syto",
    "Median_Cells_Intensity_MaxIntensity_Hoechst",
    "Median_Cells_Intensity_MaxIntensity_Mito",
    "Median_Cells_Intensity_MaxIntensity_Ph_golgi",
    "Median_Cells_Intensity_MeanIntensityEdge_Hoechst",
    "Median_Cells_Intensity_MeanIntensity_Hoechst",
    "Median_Cells_Intensity_MeanIntensity_Mito",
    "Median_Cells_Intensity_MeanIntensity_Syto",
    "Median_Cells_Intensity_MedianIntensity_ER",
    "Median_Cells_Intensity_MedianIntensity_Mito",
    "Median_Cells_Intensity_MedianIntensity_Syto",
    "Median_Cells_Intensity_MinIntensityEdge_Mito",
    "Median_Cells_Intensity_MinIntensityEdge_Syto",
    "Median_Cells_Intensity_MinIntensity_Mito",
    "Median_Cells_Intensity_MinIntensity_Syto",
    "Median_Cells_Intensity_StdIntensityEdge_ER",
    "Median_Cells_Intensity_StdIntensityEdge_Mito",
    "Median_Cells_Intensity_StdIntensityEdge_Syto",
    "Median_Cells_Intensity_StdIntensity_Hoechst",
    "Median_Cells_Intensity_StdIntensity_Mito",
    "Median_Cells_Intensity_StdIntensity_Ph_golgi",
    "Median_Cells_Intensity_StdIntensity_Syto",
    "Median_Cells_Intensity_UpperQuartileIntensity_Hoechst",
    "Median_Cells_Intensity_UpperQuartileIntensity_Mito",
    "Median_Cells_Intensity_UpperQuartileIntensity_Syto",
    "Median_Cells_RadialDistribution_FracAtD_Mito_3of4",
    "Median_Cells_RadialDistribution_FracAtD_Mito_4of4",
    "Median_Cells_RadialDistribution_FracAtD_Ph_golgi_1of4",
    "Median_Cells_RadialDistribution_FracAtD_Ph_golgi_2of4",
    "Median_Cells_RadialDistribution_FracAtD_Ph_golgi_4of4",
    "Median_Cells_RadialDistribution_MeanFrac_Mito_1of4",
    "Median_Cells_RadialDistribution_MeanFrac_Mito_2of4",
    "Median_Cells_RadialDistribution_MeanFrac_Mito_3of4",
    "Median_Cells_RadialDistribution_MeanFrac_Mito_4of4",
    "Median_Cells_RadialDistribution_MeanFrac_Ph_golgi_1of4",
    "Median_Cells_RadialDistribution_MeanFrac_Ph_golgi_2of4",
    "Median_Cells_RadialDistribution_MeanFrac_Ph_golgi_4of4",
    "Median_Cells_RadialDistribution_RadialCV_Mito_3of4",
    "Median_Cells_RadialDistribution_RadialCV_Mito_4of4",
    "Median_Cells_RadialDistribution_RadialCV_Ph_golgi_1of4",
    "Median_Cells_RadialDistribution_RadialCV_Ph_golgi_2of4",
    "Median_Cells_RadialDistribution_RadialCV_Ph_golgi_3of4",
    "Median_Cells_Texture_AngularSecondMoment_Mito_10_00",
    "Median_Cells_Texture_AngularSecondMoment_Mito_3_00",
    "Median_Cells_Texture_AngularSecondMoment_Mito_5_00",
    "Median_Cells_Texture_AngularSecondMoment_Ph_golgi_10_00",
    "Median_Cells_Texture_AngularSecondMoment_Ph_golgi_3_00",
    "Median_Cells_Texture_AngularSecondMoment_Ph_golgi_5_00",
    "Median_Cells_Texture_AngularSecondMoment_Syto_10_00",
    "Median_Cells_Texture_AngularSecondMoment_Syto_3_00",
    "Median_Cells_Texture_AngularSecondMoment_Syto_5_00",
    "Median_Cells_Texture_Contrast_ER_3_00",
    "Median_Cells_Texture_Contrast_ER_5_00",
    "Median_Cells_Texture_Contrast_Hoechst_10_00",
    "Median_Cells_Texture_Contrast_Hoechst_3_00",
    "Median_Cells_Texture_Contrast_Hoechst_5_00",
    "Median_Cells_Texture_Contrast_Mito_10_00",
    "Median_Cells_Texture_Contrast_Mito_3_00",
    "Median_Cells_Texture_Contrast_Mito_5_00",
    "Median_Cells_Texture_Contrast_Ph_golgi_10_00",
    "Median_Cells_Texture_Contrast_Ph_golgi_3_00",
    "Median_Cells_Texture_Contrast_Ph_golgi_5_00",
    "Median_Cells_Texture_Contrast_Syto_10_00",
    "Median_Cells_Texture_Contrast_Syto_3_00",
    "Median_Cells_Texture_Contrast_Syto_5_00",
    "Median_Cells_Texture_Correlation_ER_10_00",
    "Median_Cells_Texture_Correlation_ER_3_00",
    "Median_Cells_Texture_Correlation_ER_5_00",
    "Median_Cells_Texture_Correlation_Mito_10_00",
    "Median_Cells_Texture_Correlation_Mito_3_00",
    "Median_Cells_Texture_Correlation_Mito_5_00",
    "Median_Cells_Texture_Correlation_Ph_golgi_10_00",
    "Median_Cells_Texture_Correlation_Ph_golgi_3_00",
    "Median_Cells_Texture_Correlation_Ph_golgi_5_00",
    "Median_Cells_Texture_Correlation_Syto_10_00",
    "Median_Cells_Texture_Correlation_Syto_5_00",
    "Median_Cells_Texture_DifferenceEntropy_Hoechst_10_00",
    "Median_Cells_Texture_DifferenceEntropy_Mito_10_00",
    "Median_Cells_Texture_DifferenceEntropy_Mito_3_00",
    "Median_Cells_Texture_DifferenceEntropy_Mito_5_00",
    "Median_Cells_Texture_DifferenceEntropy_Ph_golgi_10_00",
    "Median_Cells_Texture_DifferenceEntropy_Ph_golgi_3_00",
    "Median_Cells_Texture_DifferenceEntropy_Ph_golgi_5_00",
    "Median_Cells_Texture_DifferenceEntropy_Syto_10_00",
    "Median_Cells_Texture_DifferenceEntropy_Syto_3_00",
    "Median_Cells_Texture_DifferenceEntropy_Syto_5_00",
    "Median_Cells_Texture_DifferenceVariance_Mito_10_00",
    "Median_Cells_Texture_DifferenceVariance_Mito_3_00",
    "Median_Cells_Texture_DifferenceVariance_Mito_5_00",
    "Median_Cells_Texture_DifferenceVariance_Ph_golgi_10_00",
    "Median_Cells_Texture_DifferenceVariance_Ph_golgi_3_00",
    "Median_Cells_Texture_DifferenceVariance_Ph_golgi_5_00",
    "Median_Cells_Texture_DifferenceVariance_Syto_10_00",
    "Median_Cells_Texture_DifferenceVariance_Syto_3_00",
    "Median_Cells_Texture_DifferenceVariance_Syto_5_00",
    "Median_Cells_Texture_Entropy_Mito_10_00",
    "Median_Cells_Texture_Entropy_Mito_3_00",
    "Median_Cells_Texture_Entropy_Mito_5_00",
    "Median_Cells_Texture_Entropy_Ph_golgi_10_00",
    "Median_Cells_Texture_Entropy_Ph_golgi_3_00",
    "Median_Cells_Texture_Entropy_Ph_golgi_5_00",
    "Median_Cells_Texture_Entropy_Syto_10_00",
    "Median_Cells_Texture_Entropy_Syto_3_00",
    "Median_Cells_Texture_Entropy_Syto_5_00",
    "Median_Cells_Texture_InfoMeas2_Ph_golgi_10_00",
    "Median_Cells_Texture_InfoMeas2_Ph_golgi_3_00",
    "Median_Cells_Texture_InfoMeas2_Ph_golgi_5_00",
    "Median_Cells_Texture_InverseDifferenceMoment_ER_10_00",
    "Median_Cells_Texture_InverseDifferenceMoment_ER_3_00",
    "Median_Cells_Texture_InverseDifferenceMoment_ER_5_00",
    "Median_Cells_Texture_InverseDifferenceMoment_Mito_10_00",
    "Median_Cells_Texture_InverseDifferenceMoment_Mito_3_00",
    "Median_Cells_Texture_InverseDifferenceMoment_Mito_5_00",
    "Median_Cells_Texture_InverseDifferenceMoment_Ph_golgi_10_00",
    "Median_Cells_Texture_InverseDifferenceMoment_Ph_golgi_3_00",
    "Median_Cells_Texture_InverseDifferenceMoment_Ph_golgi_5_00",
    "Median_Cells_Texture_InverseDifferenceMoment_Syto_10_00",
    "Median_Cells_Texture_InverseDifferenceMoment_Syto_3_00",
    "Median_Cells_Texture_InverseDifferenceMoment_Syto_5_00",
    "Median_Cells_Texture_SumAverage_Hoechst_10_00",
    "Median_Cells_Texture_SumAverage_Hoechst_3_00",
    "Median_Cells_Texture_SumAverage_Hoechst_5_00",
    "Median_Cells_Texture_SumAverage_Mito_10_00",
    "Median_Cells_Texture_SumAverage_Mito_3_00",
    "Median_Cells_Texture_SumAverage_Mito_5_00",
    "Median_Cells_Texture_SumAverage_Syto_10_00",
    "Median_Cells_Texture_SumAverage_Syto_3_00",
    "Median_Cells_Texture_SumAverage_Syto_5_00",
    "Median_Cells_Texture_SumEntropy_Mito_10_00",
    "Median_Cells_Texture_SumEntropy_Mito_3_00",
    "Median_Cells_Texture_SumEntropy_Mito_5_00",
    "Median_Cells_Texture_SumEntropy_Ph_golgi_10_00",
    "Median_Cells_Texture_SumEntropy_Ph_golgi_3_00",
    "Median_Cells_Texture_SumEntropy_Ph_golgi_5_00",
    "Median_Cells_Texture_SumEntropy_Syto_10_00",
    "Median_Cells_Texture_SumEntropy_Syto_3_00",
    "Median_Cells_Texture_SumEntropy_Syto_5_00",
    "Median_Cells_Texture_SumVariance_Hoechst_10_00",
    "Median_Cells_Texture_SumVariance_Hoechst_3_00",
    "Median_Cells_Texture_SumVariance_Hoechst_5_00",
    "Median_Cells_Texture_SumVariance_Mito_10_00",
    "Median_Cells_Texture_SumVariance_Mito_3_00",
    "Median_Cells_Texture_SumVariance_Mito_5_00",
    "Median_Cells_Texture_SumVariance_Ph_golgi_10_00",
    "Median_Cells_Texture_SumVariance_Ph_golgi_3_00",
    "Median_Cells_Texture_SumVariance_Ph_golgi_5_00",
    "Median_Cells_Texture_SumVariance_Syto_3_00",
    "Median_Cells_Texture_SumVariance_Syto_5_00",
    "Median_Cells_Texture_Variance_Hoechst_10_00",
    "Median_Cells_Texture_Variance_Hoechst_3_00",
    "Median_Cells_Texture_Variance_Hoechst_5_00",
    "Median_Cells_Texture_Variance_Mito_10_00",
    "Median_Cells_Texture_Variance_Mito_3_00",
    "Median_Cells_Texture_Variance_Mito_5_00",
    "Median_Cells_Texture_Variance_Ph_golgi_10_00",
    "Median_Cells_Texture_Variance_Ph_golgi_3_00",
    "Median_Cells_Texture_Variance_Ph_golgi_5_00",
    "Median_Cells_Texture_Variance_Syto_10_00",
    "Median_Cells_Texture_Variance_Syto_3_00",
    "Median_Cells_Texture_Variance_Syto_5_00",
    "Median_Cytoplasm_AreaShape_Area",
    "Median_Cytoplasm_AreaShape_MinFeretDiameter",
    "Median_Cytoplasm_AreaShape_MinorAxisLength",
    "Median_Cytoplasm_AreaShape_Perimeter",
    "Median_Cytoplasm_Correlation_Correlation_ER_Ph_golgi",
    "Median_Cytoplasm_Correlation_Correlation_ER_Syto",
    "Median_Cytoplasm_Correlation_Correlation_Hoechst_ER",
    "Median_Cytoplasm_Correlation_Correlation_Hoechst_Mito",
    "Median_Cytoplasm_Correlation_Correlation_Hoechst_Ph_golgi",
    "Median_Cytoplasm_Correlation_Correlation_Hoechst_Syto",
    "Median_Cytoplasm_Correlation_Correlation_Mito_ER",
    "Median_Cytoplasm_Correlation_Correlation_Mito_Ph_golgi",
    "Median_Cytoplasm_Correlation_Correlation_Mito_Syto",
    "Median_Cytoplasm_Correlation_Correlation_Syto_Ph_golgi",
    "Median_Cytoplasm_Correlation_K_ER_Syto",
    "Median_Cytoplasm_Correlation_K_Hoechst_ER",
    "Median_Cytoplasm_Correlation_K_Hoechst_Mito",
    "Median_Cytoplasm_Correlation_K_Hoechst_Ph_golgi",
    "Median_Cytoplasm_Correlation_K_Hoechst_Syto",
    "Median_Cytoplasm_Correlation_K_Mito_Hoechst",
    "Median_Cytoplasm_Correlation_K_Mito_Syto",
    "Median_Cytoplasm_Correlation_K_Ph_golgi_Syto",
    "Median_Cytoplasm_Correlation_K_Syto_ER",
    "Median_Cytoplasm_Correlation_K_Syto_Hoechst",
    "Median_Cytoplasm_Correlation_K_Syto_Mito",
    "Median_Cytoplasm_Correlation_K_Syto_Ph_golgi",
    "Median_Cytoplasm_Correlation_Manders_ER_Hoechst",
    "Median_Cytoplasm_Correlation_Manders_ER_Syto",
    "Median_Cytoplasm_Correlation_Manders_Hoechst_Syto",
    "Median_Cytoplasm_Correlation_Manders_Mito_Hoechst",
    "Median_Cytoplasm_Correlation_Manders_Mito_Syto",
    "Median_Cytoplasm_Correlation_Manders_Ph_golgi_Hoechst",
    "Median_Cytoplasm_Correlation_Manders_Ph_golgi_Syto",
    "Median_Cytoplasm_Correlation_Manders_Syto_Hoechst",
    "Median_Cytoplasm_Correlation_Overlap_ER_Syto",
    "Median_Cytoplasm_Correlation_Overlap_Hoechst_ER",
    "Median_Cytoplasm_Correlation_Overlap_Hoechst_Mito",
    "Median_Cytoplasm_Correlation_Overlap_Hoechst_Ph_golgi",
    "Median_Cytoplasm_Correlation_Overlap_Mito_Ph_golgi",
    "Median_Cytoplasm_Correlation_Overlap_Mito_Syto",
    "Median_Cytoplasm_Correlation_Overlap_Syto_Ph_golgi",
    "Median_Cytoplasm_Correlation_RWC_ER_Hoechst",
    "Median_Cytoplasm_Correlation_RWC_ER_Mito",
    "Median_Cytoplasm_Correlation_RWC_Hoechst_Mito",
    "Median_Cytoplasm_Correlation_RWC_Hoechst_Ph_golgi",
    "Median_Cytoplasm_Correlation_RWC_Hoechst_Syto",
    "Median_Cytoplasm_Correlation_RWC_Mito_Hoechst",
    "Median_Cytoplasm_Correlation_RWC_Mito_Syto",
    "Median_Cytoplasm_Correlation_RWC_Ph_golgi_Hoechst",
    "Median_Cytoplasm_Correlation_RWC_Ph_golgi_Mito",
    "Median_Cytoplasm_Correlation_RWC_Ph_golgi_Syto",
    "Median_Cytoplasm_Correlation_RWC_Syto_Hoechst",
    "Median_Cytoplasm_Correlation_RWC_Syto_Mito",
    "Median_Cytoplasm_Granularity_1_Mito",
    "Median_Cytoplasm_Granularity_1_Ph_golgi",
    "Median_Cytoplasm_Granularity_1_Syto",
    "Median_Cytoplasm_Granularity_2_ER",
    "Median_Cytoplasm_Granularity_2_Mito",
    "Median_Cytoplasm_Granularity_2_Ph_golgi",
    "Median_Cytoplasm_Granularity_3_ER",
    "Median_Cytoplasm_Granularity_3_Mito",
    "Median_Cytoplasm_Granularity_3_Ph_golgi",
    "Median_Cytoplasm_Granularity_3_Syto",
    "Median_Cytoplasm_Granularity_4_Mito",
    "Median_Cytoplasm_Granularity_4_Ph_golgi",
    "Median_Cytoplasm_Granularity_5_Mito",
    "Median_Cytoplasm_Granularity_5_Ph_golgi",
    "Median_Cytoplasm_Intensity_IntegratedIntensity_Syto",
    "Median_Cytoplasm_Intensity_LowerQuartileIntensity_Mito",
    "Median_Cytoplasm_Intensity_MADIntensity_ER",
    "Median_Cytoplasm_Intensity_MADIntensity_Hoechst",
    "Median_Cytoplasm_Intensity_MADIntensity_Mito",
    "Median_Cytoplasm_Intensity_MADIntensity_Ph_golgi",
    "Median_Cytoplasm_Intensity_MADIntensity_Syto",
    "Median_Cytoplasm_Intensity_MaxIntensityEdge_Hoechst",
    "Median_Cytoplasm_Intensity_MaxIntensityEdge_Mito",
    "Median_Cytoplasm_Intensity_MaxIntensityEdge_Ph_golgi",
    "Median_Cytoplasm_Intensity_MaxIntensityEdge_Syto",
    "Median_Cytoplasm_Intensity_MaxIntensity_Hoechst",
    "Median_Cytoplasm_Intensity_MaxIntensity_Mito",
    "Median_Cytoplasm_Intensity_MaxIntensity_Ph_golgi",
    "Median_Cytoplasm_Intensity_MaxIntensity_Syto",
    "Median_Cytoplasm_Intensity_MeanIntensityEdge_Hoechst",
    "Median_Cytoplasm_Intensity_MeanIntensity_Mito",
    "Median_Cytoplasm_Intensity_MeanIntensity_Syto",
    "Median_Cytoplasm_Intensity_MedianIntensity_ER",
    "Median_Cytoplasm_Intensity_MedianIntensity_Mito",
    "Median_Cytoplasm_Intensity_MedianIntensity_Syto",
    "Median_Cytoplasm_Intensity_MinIntensityEdge_Mito",
    "Median_Cytoplasm_Intensity_MinIntensityEdge_Syto",
    "Median_Cytoplasm_Intensity_MinIntensity_Mito",
    "Median_Cytoplasm_Intensity_MinIntensity_Syto",
    "Median_Cytoplasm_Intensity_StdIntensityEdge_Hoechst",
    "Median_Cytoplasm_Intensity_StdIntensityEdge_Mito",
    "Median_Cytoplasm_Intensity_StdIntensityEdge_Ph_golgi",
    "Median_Cytoplasm_Intensity_StdIntensityEdge_Syto",
    "Median_Cytoplasm_Intensity_StdIntensity_Hoechst",
    "Median_Cytoplasm_Intensity_StdIntensity_Mito",
    "Median_Cytoplasm_Intensity_StdIntensity_Ph_golgi",
    "Median_Cytoplasm_Intensity_StdIntensity_Syto",
    "Median_Cytoplasm_Intensity_UpperQuartileIntensity_ER",
    "Median_Cytoplasm_Intensity_UpperQuartileIntensity_Mito",
    "Median_Cytoplasm_Intensity_UpperQuartileIntensity_Syto",
    "Median_Cytoplasm_RadialDistribution_MeanFrac_ER_1of4",
    "Median_Cytoplasm_RadialDistribution_MeanFrac_ER_2of4",
    "Median_Cytoplasm_RadialDistribution_MeanFrac_Mito_1of4",
    "Median_Cytoplasm_RadialDistribution_MeanFrac_Mito_2of4",
    "Median_Cytoplasm_RadialDistribution_MeanFrac_Mito_3of4",
    "Median_Cytoplasm_RadialDistribution_MeanFrac_Mito_4of4",
    "Median_Cytoplasm_RadialDistribution_MeanFrac_Ph_golgi_1of4",
    "Median_Cytoplasm_RadialDistribution_MeanFrac_Ph_golgi_2of4",
    "Median_Cytoplasm_RadialDistribution_MeanFrac_Ph_golgi_3of4",
    "Median_Cytoplasm_RadialDistribution_MeanFrac_Ph_golgi_4of4",
    "Median_Cytoplasm_RadialDistribution_RadialCV_Mito_1of4",
    "Median_Cytoplasm_RadialDistribution_RadialCV_Mito_2of4",
    "Median_Cytoplasm_RadialDistribution_RadialCV_Mito_3of4",
    "Median_Cytoplasm_RadialDistribution_RadialCV_Mito_4of4",
    "Median_Cytoplasm_RadialDistribution_RadialCV_Ph_golgi_1of4",
    "Median_Cytoplasm_RadialDistribution_RadialCV_Ph_golgi_2of4",
    "Median_Cytoplasm_RadialDistribution_RadialCV_Ph_golgi_3of4",
    "Median_Cytoplasm_Texture_AngularSecondMoment_ER_3_00",
    "Median_Cytoplasm_Texture_AngularSecondMoment_ER_5_00",
    "Median_Cytoplasm_Texture_AngularSecondMoment_Mito_10_00",
    "Median_Cytoplasm_Texture_AngularSecondMoment_Mito_3_00",
    "Median_Cytoplasm_Texture_AngularSecondMoment_Mito_5_00",
    "Median_Cytoplasm_Texture_AngularSecondMoment_Ph_golgi_10_00",
    "Median_Cytoplasm_Texture_AngularSecondMoment_Ph_golgi_3_00",
    "Median_Cytoplasm_Texture_AngularSecondMoment_Ph_golgi_5_00",
    "Median_Cytoplasm_Texture_AngularSecondMoment_Syto_10_00",
    "Median_Cytoplasm_Texture_AngularSecondMoment_Syto_3_00",
    "Median_Cytoplasm_Texture_AngularSecondMoment_Syto_5_00",
    "Median_Cytoplasm_Texture_Contrast_ER_10_00",
    "Median_Cytoplasm_Texture_Contrast_ER_3_00",
    "Median_Cytoplasm_Texture_Contrast_ER_5_00",
    "Median_Cytoplasm_Texture_Contrast_Hoechst_10_00",
    "Median_Cytoplasm_Texture_Contrast_Hoechst_3_00",
    "Median_Cytoplasm_Texture_Contrast_Hoechst_5_00",
    "Median_Cytoplasm_Texture_Contrast_Mito_10_00",
    "Median_Cytoplasm_Texture_Contrast_Mito_3_00",
    "Median_Cytoplasm_Texture_Contrast_Mito_5_00",
    "Median_Cytoplasm_Texture_Contrast_Ph_golgi_10_00",
    "Median_Cytoplasm_Texture_Contrast_Ph_golgi_3_00",
    "Median_Cytoplasm_Texture_Contrast_Ph_golgi_5_00",
    "Median_Cytoplasm_Texture_Contrast_Syto_10_00",
    "Median_Cytoplasm_Texture_Contrast_Syto_5_00",
    "Median_Cytoplasm_Texture_Correlation_Ph_golgi_10_00",
    "Median_Cytoplasm_Texture_Correlation_Ph_golgi_3_00",
    "Median_Cytoplasm_Texture_Correlation_Ph_golgi_5_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_ER_3_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_ER_5_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_Mito_10_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_Mito_3_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_Mito_5_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_Ph_golgi_10_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_Ph_golgi_3_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_Ph_golgi_5_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_Syto_10_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_Syto_3_00",
    "Median_Cytoplasm_Texture_DifferenceEntropy_Syto_5_00",
    "Median_Cytoplasm_Texture_DifferenceVariance_Mito_10_00",
    "Median_Cytoplasm_Texture_DifferenceVariance_Mito_3_00",
    "Median_Cytoplasm_Texture_DifferenceVariance_Mito_5_00",
    "Median_Cytoplasm_Texture_DifferenceVariance_Ph_golgi_10_00",
    "Median_Cytoplasm_Texture_DifferenceVariance_Ph_golgi_3_00",
    "Median_Cytoplasm_Texture_DifferenceVariance_Ph_golgi_5_00",
    "Median_Cytoplasm_Texture_DifferenceVariance_Syto_10_00",
    "Median_Cytoplasm_Texture_DifferenceVariance_Syto_3_00",
    "Median_Cytoplasm_Texture_DifferenceVariance_Syto_5_00",
    "Median_Cytoplasm_Texture_Entropy_ER_3_00",
    "Median_Cytoplasm_Texture_Entropy_ER_5_00",
    "Median_Cytoplasm_Texture_Entropy_Mito_10_00",
    "Median_Cytoplasm_Texture_Entropy_Mito_3_00",
    "Median_Cytoplasm_Texture_Entropy_Mito_5_00",
    "Median_Cytoplasm_Texture_Entropy_Ph_golgi_10_00",
    "Median_Cytoplasm_Texture_Entropy_Ph_golgi_3_00",
    "Median_Cytoplasm_Texture_Entropy_Ph_golgi_5_00",
    "Median_Cytoplasm_Texture_Entropy_Syto_10_00",
    "Median_Cytoplasm_Texture_Entropy_Syto_3_00",
    "Median_Cytoplasm_Texture_Entropy_Syto_5_00",
    "Median_Cytoplasm_Texture_InfoMeas2_Syto_10_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_ER_10_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_ER_3_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_ER_5_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_Mito_10_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_Mito_3_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_Mito_5_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_Ph_golgi_10_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_Ph_golgi_3_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_Ph_golgi_5_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_Syto_10_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_Syto_3_00",
    "Median_Cytoplasm_Texture_InverseDifferenceMoment_Syto_5_00",
    "Median_Cytoplasm_Texture_SumAverage_ER_10_00",
    "Median_Cytoplasm_Texture_SumAverage_Mito_10_00",
    "Median_Cytoplasm_Texture_SumAverage_Mito_3_00",
    "Median_Cytoplasm_Texture_SumAverage_Mito_5_00",
    "Median_Cytoplasm_Texture_SumAverage_Syto_10_00",
    "Median_Cytoplasm_Texture_SumAverage_Syto_3_00",
    "Median_Cytoplasm_Texture_SumAverage_Syto_5_00",
    "Median_Cytoplasm_Texture_SumEntropy_Mito_10_00",
    "Median_Cytoplasm_Texture_SumEntropy_Mito_3_00",
    "Median_Cytoplasm_Texture_SumEntropy_Mito_5_00",
    "Median_Cytoplasm_Texture_SumEntropy_Ph_golgi_10_00",
    "Median_Cytoplasm_Texture_SumEntropy_Ph_golgi_3_00",
    "Median_Cytoplasm_Texture_SumEntropy_Ph_golgi_5_00",
    "Median_Cytoplasm_Texture_SumEntropy_Syto_10_00",
    "Median_Cytoplasm_Texture_SumEntropy_Syto_3_00",
    "Median_Cytoplasm_Texture_SumEntropy_Syto_5_00",
    "Median_Cytoplasm_Texture_SumVariance_Hoechst_10_00",
    "Median_Cytoplasm_Texture_SumVariance_Hoechst_3_00",
    "Median_Cytoplasm_Texture_SumVariance_Hoechst_5_00",
    "Median_Cytoplasm_Texture_SumVariance_Mito_10_00",
    "Median_Cytoplasm_Texture_SumVariance_Mito_3_00",
    "Median_Cytoplasm_Texture_SumVariance_Mito_5_00",
    "Median_Cytoplasm_Texture_SumVariance_Ph_golgi_10_00",
    "Median_Cytoplasm_Texture_SumVariance_Ph_golgi_3_00",
    "Median_Cytoplasm_Texture_SumVariance_Ph_golgi_5_00",
    "Median_Cytoplasm_Texture_SumVariance_Syto_10_00",
    "Median_Cytoplasm_Texture_SumVariance_Syto_3_00",
    "Median_Cytoplasm_Texture_SumVariance_Syto_5_00",
    "Median_Cytoplasm_Texture_Variance_Hoechst_10_00",
    "Median_Cytoplasm_Texture_Variance_Hoechst_3_00",
    "Median_Cytoplasm_Texture_Variance_Hoechst_5_00",
    "Median_Cytoplasm_Texture_Variance_Mito_10_00",
    "Median_Cytoplasm_Texture_Variance_Mito_3_00",
    "Median_Cytoplasm_Texture_Variance_Mito_5_00",
    "Median_Cytoplasm_Texture_Variance_Ph_golgi_10_00",
    "Median_Cytoplasm_Texture_Variance_Ph_golgi_3_00",
    "Median_Cytoplasm_Texture_Variance_Ph_golgi_5_00",
    "Median_Cytoplasm_Texture_Variance_Syto_10_00",
    "Median_Cytoplasm_Texture_Variance_Syto_3_00",
    "Median_Cytoplasm_Texture_Variance_Syto_5_00",
    "Median_Nuclei_AreaShape_Area",
    "Median_Nuclei_AreaShape_MajorAxisLength",
    "Median_Nuclei_AreaShape_MaxFeretDiameter",
    "Median_Nuclei_AreaShape_Perimeter",
    "Median_Nuclei_AreaShape_Solidity",
    "Median_Nuclei_Correlation_Correlation_Hoechst_Syto",
    "Median_Nuclei_Correlation_Correlation_Mito_Ph_golgi",
    "Median_Nuclei_Correlation_Correlation_Mito_Syto",
    "Median_Nuclei_Correlation_K_Mito_Ph_golgi",
    "Median_Nuclei_Correlation_K_Ph_golgi_Mito",
    "Median_Nuclei_Correlation_K_Ph_golgi_Syto",
    "Median_Nuclei_Correlation_K_Syto_Ph_golgi",
    "Median_Nuclei_Correlation_Overlap_Hoechst_Mito",
    "Median_Nuclei_Correlation_Overlap_Hoechst_Ph_golgi",
    "Median_Nuclei_Correlation_Overlap_Hoechst_Syto",
    "Median_Nuclei_Correlation_RWC_Mito_Syto",
    "Median_Nuclei_Correlation_RWC_Syto_Mito",
    "Median_Nuclei_Granularity_1_Mito",
    "Median_Nuclei_Granularity_2_Hoechst",
    "Median_Nuclei_Granularity_3_Mito",
    "Median_Nuclei_Granularity_3_Ph_golgi",
    "Median_Nuclei_Granularity_4_Mito",
    "Median_Nuclei_Intensity_LowerQuartileIntensity_Hoechst",
    "Median_Nuclei_Intensity_MADIntensity_Hoechst",
    "Median_Nuclei_Intensity_MADIntensity_Mito",
    "Median_Nuclei_Intensity_MaxIntensityEdge_Hoechst",
    "Median_Nuclei_Intensity_MaxIntensityEdge_Mito",
    "Median_Nuclei_Intensity_MaxIntensityEdge_Ph_golgi",
    "Median_Nuclei_Intensity_MaxIntensityEdge_Syto",
    "Median_Nuclei_Intensity_MaxIntensity_Hoechst",
    "Median_Nuclei_Intensity_MaxIntensity_Mito",
    "Median_Nuclei_Intensity_MaxIntensity_Ph_golgi",
    "Median_Nuclei_Intensity_MeanIntensityEdge_Hoechst",
    "Median_Nuclei_Intensity_MeanIntensity_Hoechst",
    "Median_Nuclei_Intensity_MedianIntensity_Hoechst",
    "Median_Nuclei_Intensity_MinIntensityEdge_Hoechst",
    "Median_Nuclei_Intensity_MinIntensity_Hoechst",
    "Median_Nuclei_Intensity_StdIntensityEdge_Mito",
    "Median_Nuclei_Intensity_StdIntensityEdge_Ph_golgi",
    "Median_Nuclei_Intensity_StdIntensityEdge_Syto",
    "Median_Nuclei_Intensity_StdIntensity_Mito",
    "Median_Nuclei_Intensity_StdIntensity_Ph_golgi",
    "Median_Nuclei_Intensity_UpperQuartileIntensity_Hoechst",
    "Median_Nuclei_RadialDistribution_FracAtD_Ph_golgi_3of4",
    "Median_Nuclei_RadialDistribution_MeanFrac_Mito_1of4",
    "Median_Nuclei_RadialDistribution_MeanFrac_Mito_2of4",
    "Median_Nuclei_RadialDistribution_MeanFrac_Mito_3of4",
    "Median_Nuclei_RadialDistribution_MeanFrac_Mito_4of4",
    "Median_Nuclei_RadialDistribution_MeanFrac_Ph_golgi_2of4",
    "Median_Nuclei_RadialDistribution_MeanFrac_Ph_golgi_3of4",
    "Median_Nuclei_RadialDistribution_MeanFrac_Ph_golgi_4of4",
    "Median_Nuclei_Texture_AngularSecondMoment_Hoechst_10_00",
    "Median_Nuclei_Texture_AngularSecondMoment_Hoechst_3_00",
    "Median_Nuclei_Texture_AngularSecondMoment_Hoechst_5_00",
    "Median_Nuclei_Texture_AngularSecondMoment_Mito_3_00",
    "Median_Nuclei_Texture_AngularSecondMoment_Mito_5_00",
    "Median_Nuclei_Texture_Contrast_Hoechst_3_00",
    "Median_Nuclei_Texture_Contrast_Hoechst_5_00",
    "Median_Nuclei_Texture_Contrast_Mito_10_00",
    "Median_Nuclei_Texture_Contrast_Mito_3_00",
    "Median_Nuclei_Texture_Contrast_Mito_5_00",
    "Median_Nuclei_Texture_Contrast_Ph_golgi_10_00",
    "Median_Nuclei_Texture_Contrast_Ph_golgi_3_00",
    "Median_Nuclei_Texture_Contrast_Ph_golgi_5_00",
    "Median_Nuclei_Texture_Correlation_Ph_golgi_3_00",
    "Median_Nuclei_Texture_DifferenceEntropy_Hoechst_10_00",
    "Median_Nuclei_Texture_DifferenceEntropy_Hoechst_3_00",
    "Median_Nuclei_Texture_DifferenceEntropy_Hoechst_5_00",
    "Median_Nuclei_Texture_DifferenceEntropy_Mito_10_00",
    "Median_Nuclei_Texture_DifferenceEntropy_Mito_3_00",
    "Median_Nuclei_Texture_DifferenceEntropy_Mito_5_00",
    "Median_Nuclei_Texture_DifferenceEntropy_Ph_golgi_3_00",
    "Median_Nuclei_Texture_DifferenceEntropy_Ph_golgi_5_00",
    "Median_Nuclei_Texture_DifferenceVariance_Hoechst_10_00",
    "Median_Nuclei_Texture_DifferenceVariance_Hoechst_3_00",
    "Median_Nuclei_Texture_DifferenceVariance_Hoechst_5_00",
    "Median_Nuclei_Texture_DifferenceVariance_Mito_3_00",
    "Median_Nuclei_Texture_DifferenceVariance_Mito_5_00",
    "Median_Nuclei_Texture_DifferenceVariance_Ph_golgi_10_00",
    "Median_Nuclei_Texture_DifferenceVariance_Ph_golgi_3_00",
    "Median_Nuclei_Texture_DifferenceVariance_Ph_golgi_5_00",
    "Median_Nuclei_Texture_Entropy_Hoechst_3_00",
    "Median_Nuclei_Texture_Entropy_Hoechst_5_00",
    "Median_Nuclei_Texture_Entropy_Mito_10_00",
    "Median_Nuclei_Texture_Entropy_Mito_3_00",
    "Median_Nuclei_Texture_Entropy_Mito_5_00",
    "Median_Nuclei_Texture_InfoMeas2_Hoechst_10_00",
    "Median_Nuclei_Texture_InfoMeas2_Mito_10_00",
    "Median_Nuclei_Texture_InfoMeas2_Ph_golgi_10_00",
    "Median_Nuclei_Texture_InfoMeas2_Ph_golgi_3_00",
    "Median_Nuclei_Texture_InfoMeas2_Ph_golgi_5_00",
    "Median_Nuclei_Texture_InverseDifferenceMoment_Hoechst_10_00",
    "Median_Nuclei_Texture_InverseDifferenceMoment_Hoechst_3_00",
    "Median_Nuclei_Texture_InverseDifferenceMoment_Hoechst_5_00",
    "Median_Nuclei_Texture_InverseDifferenceMoment_Mito_10_00",
    "Median_Nuclei_Texture_InverseDifferenceMoment_Mito_3_00",
    "Median_Nuclei_Texture_InverseDifferenceMoment_Mito_5_00",
    "Median_Nuclei_Texture_SumAverage_Hoechst_10_00",
    "Median_Nuclei_Texture_SumAverage_Hoechst_3_00",
    "Median_Nuclei_Texture_SumAverage_Hoechst_5_00",
    "Median_Nuclei_Texture_SumEntropy_Mito_10_00",
    "Median_Nuclei_Texture_SumEntropy_Mito_3_00",
    "Median_Nuclei_Texture_SumEntropy_Mito_5_00",
    "Median_Nuclei_Texture_SumEntropy_Ph_golgi_10_00",
    "Median_Nuclei_Texture_SumEntropy_Ph_golgi_3_00",
    "Median_Nuclei_Texture_SumEntropy_Ph_golgi_5_00",
    "Median_Nuclei_Texture_SumVariance_Mito_10_00",
    "Median_Nuclei_Texture_SumVariance_Mito_3_00",
    "Median_Nuclei_Texture_SumVariance_Mito_5_00",
    "Median_Nuclei_Texture_SumVariance_Ph_golgi_10_00",
    "Median_Nuclei_Texture_SumVariance_Ph_golgi_3_00",
    "Median_Nuclei_Texture_SumVariance_Ph_golgi_5_00",
    "Median_Nuclei_Texture_Variance_Mito_10_00",
    "Median_Nuclei_Texture_Variance_Mito_3_00",
    "Median_Nuclei_Texture_Variance_Mito_5_00",
    "Median_Nuclei_Texture_Variance_Ph_golgi_10_00",
    "Median_Nuclei_Texture_Variance_Ph_golgi_3_00",
    "Median_Nuclei_Texture_Variance_Ph_golgi_5_00",
]

# Calculate XTICKS for the default feature set
x = 1
XTICKS = [x]
for comp in ["Median_Cytoplasm", "Median_Nuclei"]:
    for idx, p in enumerate(ACT_PROF_FEATURES[x:], 1):
        if p.startswith(comp):
            XTICKS.append(idx + x)
            x += idx
            break
XTICKS.append(len(ACT_PROF_FEATURES))


def is_close(a: float, b: float, abs_tol: float = 1e-6) -> bool:
    return abs(a - b) < abs_tol


def meta_data(df: pd.DataFrame) -> List[str]:
    """Returns the list of columns in the DataFrame that do *not* contain Cell Painting data."""
    return [x for x in df if not x.startswith("Median_")]


def feature_data(df: pd.DataFrame) -> List[str]:
    """Returns the list of columns in the DataFrame that *do* contain Cell Painting data."""
    return [x for x in df if x.startswith("Median_")]


def profile_sim(
    prof1: Iterable[float],
    prof2: Iterable[float],
) -> float:
    """Calculates the similarity of two activity_profiles of the same length.
    The profiles are compared by distance correlation
    ``scipy.spatial.distance.correlation()`` (same as Pearson correlation).
    The profile values are capped to +/-25.0 (np.clip() function).

    Parameters:
    ===========
        prof1: The first profile to compare.
        prof2: The second profile to compare.
            The two profiles have to be of equal length.


    Returns:
    ========
        Similarity value between 0.0 .. 1.0 (0.0 being very dissimilar and 1.0 identical)."""

    assert len(prof1) == len(
        prof2
    ), "Activity Profiles must have the same length to be compared."

    if not isinstance(prof1, np.ndarray):
        prof1 = np.array(prof1)
    prof1 = np.clip(prof1, -25.0, 25.0)
    if not isinstance(prof2, np.ndarray):
        prof2 = np.array(prof2)
    prof2 = np.clip(prof2, -25.0, 25.0)
    result = 1 - dist.correlation(prof1, prof2)
    if np.isnan(result) or result < 0.0:
        result = 0.0
    return result


def well_id_similarity(df: pd.DataFrame, well_id1: str, well_id2: str) -> float:
    """Calculate the similarity of the activity profiles from two compounds
    (identified by `Well_Id`). Returns value between 0 .. 1"""
    act1 = df[df["Well_Id"] == well_id1][ACT_PROF_FEATURES].values[0]
    act2 = df[df["Well_Id"] == well_id2][ACT_PROF_FEATURES].values[0]
    return round(profile_sim(act1, act2), 3)


def find_similar(
    df: pd.DataFrame,
    act_profile: Iterable[float],
    cutoff=75.0,
    max_num=5,
    features=ACT_PROF_FEATURES,
):
    """Filter the dataframe for activity profiles similar to the given one.
    `cutoff` gives the similarity threshold in percent, default is 75.
    The calculated similarity is added to the result DataFrame as a new column.
    The similarity value is given in percent (0.0 .. 100.0) in this case.

    Parameters:
    ===========
    df: pd.DataFrame
        The dataframe to filter.
    act_profile: Iterable[float]
        The activity profile to compare to.
    cutoff: float
        The similarity threshold.
    max_num: int
        The maximum number of results to return.
    features: List[str]
        The features to use for the similarity calculation.

    Returns a Pandas DF with the most similar entries (similarity in percent) or None when no similars are found."""

    act_features = features.copy()
    assert len(act_features) > 0
    decimals = {"Similarity": 1}
    if not isinstance(act_profile, np.ndarray):
        act_profile = np.array(act_profile)

    result = df.copy()
    # Pandas black belt!! :
    result["Similarity"] = result[act_features].apply(
        lambda x: profile_sim(
            x,
            act_profile,
        )
        * 100.0,
        axis=1,
    )
    result = result[result["Similarity"] >= cutoff]
    if len(result) == 0:
        return None
    result = result.sort_values("Similarity", ascending=False).head(max_num)
    result = result.round(decimals)
    return result


@functools.lru_cache
def get_func_cluster_names(prefix="") -> Optional[List[str]]:
    """Extract the cluster names from the median profile file names.
    If a `prefix` is given, it will be put in front of the names.
    Returns an alphabetically sorted list of cluster names."""
    # print(str(Path(__file__).absolute().))
    mask_files = glob(op.join(OUTPUT_DIR, "med_prof_*.tsv"))
    clusters = sorted([prefix + op.basename(x)[9:-4] for x in mask_files])
    if len(clusters) == 0:
        print(f"No clusters found in {OUTPUT_DIR}.")
        return None
    return clusters


@functools.lru_cache
def get_func_cluster_parameters(cluster: str, include_well_id=True) -> pd.DataFrame:
    """Extract the cluster parameters from the median profile files.

    Returns:
        a DataFrame WITH THE (artificial) Well_Id of the cluster
        AND the parameters (default).
        Set `include_well_id=False` to get the parameters without the Well_Id.

    Raises:
        ValueError: when the ClusterMaskDir is not set in the config
        FileNotFoundError: when the cluster parameter file is not found."""
    parm_file = op.join(OUTPUT_DIR, f"med_prof_{cluster}.tsv")
    try:
        cl_parms = pd.read_csv(parm_file, sep="\t")
    except FileNotFoundError:
        raise FileNotFoundError(f"Cluster {cluster} not found in {OUTPUT_DIR}.")
    if not include_well_id:
        cl_parms = cl_parms.drop("Well_Id", axis=1)
    return cl_parms


def add_func_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Add the similarities to the functional clusters to the dataframe.

    Raises:
        ValueError: when no functional cluster definitions are found.
        FloatingPointError: when the similarity value could not be compared for equality.
            This is a safe guard, because comparing floats for equality is not reliable.

    Returns:
        The dataframe with the functional cluster similarities added."""

    def calc_sim(series, prof1, parameters):
        prof2 = series[parameters].values.astype("float64")
        result = round(100 * profile_sim(prof1, prof2), 1)
        return result

    func_clusters = get_func_cluster_names(prefix="")
    if func_clusters is None:
        raise ValueError("No functional clusters found.")

    result = df.copy()
    for cl in func_clusters:
        med_prof = pd.read_csv(op.join(OUTPUT_DIR, f"med_prof_{cl}.tsv"), sep="\t")
        cl_feat = sorted([x for x in med_prof.keys() if x.startswith("Median_")])
        prof1 = med_prof[cl_feat].values[0]
        assert len(cl_feat) == len(prof1)

        # [1] This line only works when the dataset is a Pandas DF.
        # If it is a Dask DF all Clusters get the same similarity value (of Uncoupler)
        result[f"Cluster_{cl}"] = result.apply(
            lambda s: calc_sim(s, prof1, cl_feat), axis=1
        )

    # Find the cluster with the highest Sim for each compound:
    clusters = [f"Cluster_{cl}" for cl in func_clusters]
    most_sim = {"Well_Id": [], "Cluster_High": [], "Cluster_Sim": []}
    for _, rec in result.iterrows():
        sim = rec[clusters].max()
        for cl in clusters:
            if is_close(rec[cl], sim):
                break
        else:
            # Fail-safe for comparing floats for equality.
            raise FloatingPointError(f"Could not find Sim {sim}.")
        most_sim["Well_Id"].append(rec["Well_Id"])
        most_sim["Cluster_High"].append(cl[8:])
        most_sim["Cluster_Sim"].append(sim)

    result = result.merge(pd.DataFrame(most_sim), on="Well_Id", how="left")
    return result


def cluster_features(df: pd.DataFrame, fraction: float):
    """The list of parameters that defines a cluster.

    Parameters:
    ===========
    df: pd.DataFrame
        The dataframe to select the features from.
    fraction: float
        The fraction of feature values that need to point in the same direction
        in order to be selected for the cluster.

    Returns: a list of selected feature names.
    """
    df_len = len(df)
    result = []
    for feat in ACT_PROF_FEATURES:
        count_plus = int((df[feat] >= 0.0).sum())
        count_minus = int((df[feat] < 0.0).sum())
        value = max(count_plus, count_minus) / df_len
        if value >= fraction:
            result.append(feat)
    return result


def remaining_features(cl_feat: Iterable[str]) -> List[str]:
    """Returns the list of features after removing the cluster-defining features from the full profile."""
    feat_set = set(cl_feat)
    result = [x for x in ACT_PROF_FEATURES if x not in feat_set]
    return result


def calc_median_profile(
    df: pd.DataFrame, cl_feat: List[str], cl_name: Union[str] = None
) -> pd.DataFrame:
    """Calculate the median profile of a cluster.

    Parameters:
    ===========
    df: pd.DataFrame
        The dataframe containing the cluster measurements.
    cl_feat: List[str]
        The list of features to use for the median profile.
    cl_name: Union[str]
        The optional name of the cluster.
        If given, the name will be used as the Well_Id.
        If not given, the Well_Id will be set to "Unknown".

    Returns: A DataFrame with the median profile and the cluster name as Well_Id.
    """
    cl_name = cl_name if cl_name is not None else "Unknown"
    med_prof = df[cl_feat].median().values
    df_mp = pd.DataFrame(data=(med_prof,), columns=cl_feat)
    df_mp["Well_Id"] = cl_name
    return df_mp


def heat_mpl(
    df,
    id_prop="Compound_Id",
    cmap="bwr",
    show=True,
    sort_parm=False,
    **kwargs,
):
    """Generates a heatmap of the Cell Painting profiles.

    Parameters:
    ===========
    df: pd.DataFrame
        The dataframe to plot.
    id_prop: str
        The column to use for labeling the rows in the plot.
    cmap: str
        The colormap to use for the heatmap (default: "bwr").
    show: bool
        Whether to show the plot or not (default: True).
        When False, the plot is saved to disk (see kwarg `save_to_file`)

    Keyword arguments:
    ==================
    colorbar: bool
        Whether to show the color bar or not (default: True)
    biosim: bool
        Whether to show the biological similarity (default: False)
    show_ind: bool
        Whether to show the Induction (Activity) or not (default: False)
    color_range: int
        The value used for the color range legend (default: 15)
    img_size: Optional[int]
        The size of the image (default: None)
    features: List[str]
        The features to use for the heatmap.
    save_to_file: Union[str, List[str]]
        Save the plot as file, requires `show`=False (default: "heatmap.png")
        A single file name or a list of file names can be given
        (e.g. when the plot should be saved in multiple formats).
    rcparams: dict
        Parameters mapped to matplotlib.rcParams
    """
    # not assigned to individual variables:
    #   colorbar
    biosim = kwargs.get("biosim", False)
    show_ind = kwargs.get("show_ind", False)
    color_range = kwargs.get("color_range", 15)
    img_size = kwargs.get("img_size", None)
    features = kwargs.get("features", None)
    save_to_file = kwargs.get("save_to_file", "heatmap.png")

    if features is None:
        features = ACT_PROF_FEATURES

    # Re-calculate XTICKS for non-default parameter sets
    if len(features) == len(ACT_PROF_FEATURES):
        xticks = XTICKS  # global var
    else:
        print("  - Re-calculating xticks...")
        # get positions of the compartments in the list of features
        x = 1
        xticks = [x]
        for comp in ["Median_Cytoplasm", "Median_Nuclei"]:
            for idx, p in enumerate(features[x:], 1):
                if p.startswith(comp):
                    xticks.append(idx + x)
                    x += idx
                    break
        xticks.append(len(features))

    df_len = len(df)
    if img_size is None:  # set defaults when no img_size is given
        if show:
            img_size = 15.0
            if biosim:
                img_size += 0.5
            if show_ind:
                img_size += 0.5
            if id_prop == "Well_Id":
                img_size += 1.0
        else:
            img_size = 19.0
    plt.style.use("seaborn-white")
    plt.style.use("seaborn-pastel")
    plt.style.use("seaborn-talk")
    plt.rcParams["axes.labelsize"] = 25
    if "rcparams" in kwargs:
        plt.rcParams.update(kwargs["rcparams"])

    if df_len == 1:
        height = 0.5
    elif df_len == 2:
        height = 2.0
    else:
        height = 1.1 + 0.35 * df_len
    plt.rcParams["figure.figsize"] = (img_size, height)
    plt.rcParams["axes.labelsize"] = 25
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 15
    fs_text = 18

    y_labels = []
    fp_list = []
    max_val = color_range  # using a fixed color range now
    min_val = -color_range
    ylabel_templ = "{}{}{}"
    ylabel_bs = ""
    ylabel_ind = ""
    id_prop_list = []
    fp = []
    for ctr, (_, rec) in enumerate(df.iterrows()):
        parm_list = features
        fp = [rec[x] for x in features]
        fp_view = [rec[x] for x in parm_list]
        fp_list.append(fp_view)
        id_prop_list.append(rec[id_prop])
        if biosim:
            if ctr == 0:
                prof_ref = fp
                ylabel_bs = "   --  |  "
            else:
                sim = profile_sim(prof_ref, fp) * 100
                ylabel_bs = "{:3.0f} |  ".format(sim)
        if show_ind:
            ylabel_ind = "{:3.0f} |  ".format(rec["Activity"])

        ylabel = ylabel_templ.format(ylabel_bs, ylabel_ind, rec[id_prop])
        y_labels.append(ylabel)

    # invert y axis:
    y_labels = y_labels[::-1]
    fp_list = fp_list[::-1]
    Z = np.asarray(fp_list)
    plt.xticks(xticks)
    plt.yticks(np.arange(df_len) + 0.5, y_labels)
    plt.pcolor(Z, vmin=min_val, vmax=max_val, cmap=cmap)
    plt.text(
        xticks[1] // 2, -1.1, "Cells", horizontalalignment="center", fontsize=fs_text
    )
    plt.text(
        xticks[1] + ((xticks[2] - xticks[1]) // 2),
        -1.1,
        "Cytoplasm",
        horizontalalignment="center",
        fontsize=fs_text,
    )
    plt.text(
        xticks[2] + ((xticks[3] - xticks[2]) // 2),
        -1.1,
        "Nuclei",
        horizontalalignment="center",
        fontsize=fs_text,
    )
    if kwargs.get("colorbar", True) and len(df) > 3:
        plt.colorbar()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        if not isinstance(save_to_file, list):
            save_to_file = [save_to_file]
        for fn in save_to_file:
            plt.savefig(fn, bbox_inches="tight")
