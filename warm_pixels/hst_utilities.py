"""Utility functions etc. for hst_warm_pixels.py"""

import argparse
from pathlib import Path

import numpy as np
from astropy.time import Time

output_path = Path("output")
cache_path = Path("cache")

# ========
# Input parameters
# ========
# Number of pixels for each trail, not including the warm pixel itself
trail_length = 12  # 9
# The minimum fraction of images in which a warm pixel must be present.
# fraction_present = 0.9
fraction_present = 2 / 3
# Stacking bins (see generate_stacked_lines_from_bins() in pixel_lines.py)
n_row_bins = 5
n_flux_bins = 10
n_background_bins = 1

#   Richard bins
flux_bins = np.array(
    [
        100.000,
        177.475,
        314.972,
        558.996,
        992.075,
        1760.68,
        3124.76,
        5545.66,
        9842.13,
        17467.3,
        58982.4,
    ]
)
#   Middle flux bins
# flux_bins = np.logspace(np.log10(2e2), np.log10(1e4), n_flux_bins + 1)


# ========
# Constants
# ========
# Julian dates
date_acs_launch = 2452334.5  # ACS launched, SM3B, 01 March 2002
date_T_change = 2453920.0  # Temperature changed, 03 July 2006
date_side2_fail = 2454128.0  # ACS stopped working, 27 January 2007
date_sm4_repair = 2454968.0  # ACS repaired, SM4, 16 May 2009
# Convert to days since ACS launch
day_T_change = date_T_change - date_acs_launch
day_side2_fail = date_side2_fail - date_acs_launch
day_sm4_repair = date_sm4_repair - date_acs_launch


def dec_yr_to_jd(dates):
    """Convert one or more decimal-year dates to Julian dates."""
    time = Time(dates, format="decimalyear")
    time.format = "jd"
    return time.value


def jd_to_dec_yr(dates):
    """Convert one or more Julian dates to decimal-year dates."""
    time = Time(dates, format="jd")
    time.format = "decimalyear"
    return time.value


def dataset_list_saved_density_evol(list_name, quadrants, use_corrected=False):
    """Return the file path for the saved density data for a dataset list."""
    if use_corrected:
        suffix = "_cor"
    else:
        suffix = ""
    return output_path / f"density_evol_{list_name}_{''.join(map(str, quadrants))}{suffix}.npz"



