""" 
Example: Find warm pixels in an image from the Hubble Space Telescope (HST) 
Advanced Camera for Surveys (ACS) instrument.

A small patch of the image is plotted with the warm pixels marked with red Xs.
"""

import numpy as np
import pytest
import os
import sys
import matplotlib.pyplot as plt

from pixel_lines import PixelLine, PixelLineCollection
from warm_pixels import find_warm_pixels


import autoarray as aa

# Load the HST ACS image
dataset_path = os.path.join(path, "data/")
file_path = dataset_path + "jc0a01h8q_raw.fits"
bias_path = dataset_path + "25b1734qj_bia.fits"
image = aa.acs.ImageACS.from_fits(
    file_path=file_path,
    quadrant_letter="A",
    bias_path=bias_path,
    bias_subtract_via_prescan=True,
).native

# Extract an example patch of the full image
row_start, row_end, column_start, column_end = -300, -100, -300, -100
image = image[row_start:row_end, column_start:column_end]

# Find the warm pixel trails and store in a line collection object
warm_pixels = PixelLineCollection(lines=find_warm_pixels(image=image))

print("Found %d warm pixels" % warm_pixels.n_lines)

# Plot the image and the found warm pixels
plt.figure()
im = plt.imshow(X=image, aspect="equal", vmin=0, vmax=500)
plt.scatter(
    warm_pixels.locations[:, 1],
    warm_pixels.locations[:, 0],
    c="r",
    marker="x",
    s=4,
    linewidth=0.2,
)
plt.colorbar(im)
plt.axis("off")
save_path = dataset_path + "find_warm_pixels.png"
plt.savefig(save_path, dpi=400)
plt.close()
print("Saved", save_path)
