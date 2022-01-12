#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for analysis of the Cell Painting Assay data.
"""

# TODO: include code for matplotlib heatmap

import functools
from glob import glob
import os.path as op
from typing import Iterable, Optional, List

# import sys

import pandas as pd
import numpy as np

import scipy.spatial.distance as dist

MODULE_DIR = op.dirname(op.abspath(__file__))
OUTPUT_DIR = op.join(MODULE_DIR, "output")


ACT_PROF_PARAMETERS = [
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


def is_close(a: float, b: float, abs_tol: float = 1e-6) -> bool:
    return abs(a - b) < abs_tol


def meta_data(df: pd.DataFrame) -> List[str]:
    """Returns the list of columns in the DataFrame that do *not* contain Cell Painting data."""
    return [x for x in df if not x.startswith("Median_")]


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
    act1 = df[df["Well_Id"] == well_id1][ACT_PROF_PARAMETERS].values[0]
    act2 = df[df["Well_Id"] == well_id2][ACT_PROF_PARAMETERS].values[0]
    return round(profile_sim(act1, act2), 3)


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
