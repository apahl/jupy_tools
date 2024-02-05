#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Altair style with larger fonts."""

import altair as alt

def larger_fonts():
    return {
        'config': {
            'axis': {
                'labelFontSize': 14,
                'titleFontSize': 16
            },
            'legend': {
                'titleFontSize': 16,
                'labelFontSize': 14
            },
            'title': {
                'fontSize': 18,
                'subtitleFontSize': 16
            }
        }
    }

# register the custom theme under a chosen name
alt.themes.register('larger_fonts', larger_fonts);

# enable the newly registered theme
alt.themes.enable('larger_fonts');
