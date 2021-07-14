"""
WIP
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pixel_lines import PixelLine, PixelLineCollection
from warm_pixels import find_warm_pixels

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path, "../PyAutoArray/"))
import autoarray as aa


# ========
# Image datasets
# ========
dataset_root = os.path.join(path, "../hst_acs_datasets/")
datasets_all = [
    # Aidan
    # Before June 2006
    # After June 2006
    # Richard
    # Before June 2006
    # After June 2006
    ### unsorted
    # "01_2003",
    # "01_2015",
    # "01_2016",
    # "01_2019",
    # "01_2020",
    # "02_2014",
    # "02_2018",
    # "03_2020",
    # "04_2005",
    # "04_2006",
    # "04_2011",
    # "04_2013",
    # "04_2014",
    # "04_2017",
    # "04_2019",
    # "04_2020",
    # "05_2004",
    # "05_2005",
    # "05_2010",
    # "05_2012",
    # "05_2016",
    # "06_2015",
    # "06_2017",
    # "07_2006",
    # "07_2019",
    # "07_2019_2",
    # "07_2020",
    # "08_2017",
    # "08_2018",
    # "08_2020_1",
    # "08_2020_2",
    # "09_2005",
    # "09_2015",
    # "09_2016",
    # "09_2020",
    # "09_2020_2",
    # "10_2004",
    # "10_2017",
    # "10_2019",
    # "11_2005",
    # "11_2019",
    # "11_2019_2",
    # "12_2004",
    # "12_2018",
    # "12_2019",
    # "12_2019_2",
    # "12_2020",
    # "candels2013a",
    # "candels2013b",
    # "early",
    # "huff_spt606a",
    # "huff_spt606b",
    # "huff_spt606c",
    # "huff_spt606d",
    # "huff_spt606e",
    # "huff_spt606f",
    # "huff_spt606g",
    # "huff_spt606h",
    # "huff_spt814a",
    # "huff_spt814b",
    # "huff_spt814c",
    # "late",
    # "later",
    # "longSNe1",
    # "longSNe2",
    # "longSNe3",
    # "longSNe4",
    # "longSNe5",
    # "longSNe6",
    # "middle1",
    # "middle2",
    # "obama",
    # "obama2",
    # "richmassey60484",
    # "richmassey60485",
    # "richmassey60486",
    # "richmassey60487",
    # "richmassey60488",
    # "richmassey60489",
    # "richmassey60490",
    # "richmassey60491",
    # "richmassey60492",
    # "richmassey60493",
    # "richmassey60494",
    # "richmassey60680",
    # "richmassey61092",
    # "richmassey61093",
    # "richmassey72698",
    # "richmassey72699",
    # "richmassey72700",
    # "richmassey72701",
    # "richmassey72702",
    # "richmassey72703",
    # "richmassey72704",
    # "shortSNe0",
    # "shortSNe1",
    # "shortSNe2",
    # "shortSNe3",
    # "shortSNe4",
    # "shortSNe5",
    # "shortSNe6",
    # "shortSNe7",
    # "shortSNe8",
    # "shortSNe9",
    # "shortSNeA",
    # "sm41",
    # "sm42",
    # "sm43",
    # "sm44",
    # "sm45",
    # "ten1a",
    # "ten1b",
    # "ten2a",
    # "ten2b",
    # "ten3",
    # "ten4",
]
datasets_test = ["12_2020"]


# ========
# Functions etc
# ========
class Dataset(object):
    """WIP"""

    def __init__(self, name):
        self.name = name
        self.path = dataset_root + self.name + "/"

        # Image file paths
        files = os.listdir(dataset_root + self.name)
        self.image_names = [f[:-5] for f in files if f[-9:] == "_raw.fits"]
        self.image_paths = [self.path + name + ".fits" for name in self.image_names]
        self.n_images = len(self.image_names)

        # Bias file path
        self.bias_name = [f[:-5] for f in files if f[-9:] == "_bia.fits"][0]
        self.bias_path = self.path + self.bias_name + ".fits"

        # Save paths
        self.saved_lines = self.path + "saved_lines"
        self.saved_consistent_lines = self.path + "saved_consistent_lines"
        self.saved_stacked_lines = self.path + "saved_stacked_lines"
        self.saved_stacked_info = self.path + "saved_stacked_info"


def plot_warm_pixels(image, warm_pixels, save_path=None):
    """WIP"""
    # Plot the image and the found warm pixels
    plt.figure()
    im = plt.imshow(X=image, aspect="equal", vmin=0, vmax=500)
    plt.scatter(
        warm_pixels.locations[:, 1],
        warm_pixels.locations[:, 0],
        marker=".",
        c="r",
        edgecolor="none",
        s=0.1,
        alpha=0.7,
    )
    plt.colorbar(im)
    plt.axis("off")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=800)
        # print("Saved", save_path[-40:])
        plt.close()


# ========
# Main
# ========
if __name__ == "__main__":
    # ========
    # Find and stack warm pixels in each dataset
    # ========
    for name in datasets_test:
        dataset = Dataset(name)

        # Skip if already found and saved
        if False:
            continue

        print('\nDataset "%s", %d images' % (name, dataset.n_images))

        # Initialise the collection of warm pixel trails
        warm_pixels = PixelLineCollection()


        # ========
        # Warm pixels in each image
        # ========
        if  True:
            print("1. Find possible warm pixels")
            # Find the warm pixels in each image
            for i_image in range(dataset.n_images):
                image_path = dataset.image_paths[i_image]
                image_name = dataset.image_names[i_image]
                print(
                    "  %s (%d of %d): " % (image_name, i_image + 1, dataset.n_images),
                    end="",
                    flush=True,
                )

                # Load the image
                image = aa.acs.ImageACS.from_fits(
                    file_path=image_path,
                    quadrant_letter="A",
                    bias_path=dataset.bias_path,
                    bias_subtract_via_prescan=True,
                ).native

                date = 2400000.5 + image.header.modified_julian_date

                # Find the warm pixel trails
                new_warm_pixels = find_warm_pixels(
                    image=image, origin=image_name, date=date
                )
                print("Found %d possible warm pixels" % len(new_warm_pixels))

                # Plot
                plot_warm_pixels(
                    image,
                    PixelLineCollection(new_warm_pixels),
                    save_path=dataset.path + image_name,
                )

                # Add them to the collection
                warm_pixels.append(new_warm_pixels)

            # Save
            warm_pixels.save(dataset.saved_lines)


        # ========
        # Consistent warm pixels in the set
        # ========
        if True:
            print("2. Find consistent warm pixels")

            # Load
            warm_pixels = PixelLineCollection()
            warm_pixels.load(dataset.saved_lines)

            # Find the warm pixels present in at least 2/3 of the images
            consistent_lines = warm_pixels.find_consistent_lines(fraction_present=2 / 3)
            print(
                "Found %d consistent warm pixels out of %d possibles"
                % (len(consistent_lines), warm_pixels.n_lines)
            )

            # Extract the consistent warm pixels
            warm_pixels.lines = warm_pixels.lines[consistent_lines]

            # Save
            warm_pixels.save(dataset.saved_consistent_lines)
