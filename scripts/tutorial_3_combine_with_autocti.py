import json
from pathlib import Path
import pickle

from autocti.dataset_1d.dataset_1d.dataset_1d import Dataset1D

import warm_pixels as wp

"""
We assume our data is in a directory parallel to warm_pixels called 'hst_acs_datasets'
"""
data_directory = Path.cwd().parent / "hst_acs_datasets"

"""
We can create an object to access all datasets in that directory
"""
source = wp.Source(
    data_directory
)

"""
We can get a list of datasets from that source
"""
datasets = list(source)

"""
Each dataset comprises several images captured on a given date
"""
dataset = datasets[0]

"""
We can get the images from a dataset
"""
images = dataset.images
image = images[0]

"""
Each image has 4 quadrants which can be addressed by letter
"""
quadrant_a = image.quadrant("A")

"""
We can get an array for that quadrant
"""
array = quadrant_a.array()

"""
We can search for warm pixels by making a WarmPixels object from our dataset
"""
warm_pixels = wp.WarmPixels(
    datasets=datasets,
)

"""
Groups of quadrants with one item for each dataset.
"""
groups = warm_pixels.all_groups()

"""
A list of groups for the first dataset. There is one group for each combination
of quadrants. By default there is only one group with quadrants ABCD
"""
first_dataset_groups = groups[0]
group = first_dataset_groups[0]

"""
Warm pixel lines are extracted and grouped into bins by their attributes.
1) Find warm pixels
2) Extract the CTI trail corresponding to those warm pixels
3) Keep only those lines which appear consistently across multiple images
4) Stack warm pixels into bins with similar CCD row, flux, background and date
"""
stacked_lines = group.stacked_lines()

"""
Each stacked line has a dictionary representation
"""
stacked_line_dicts = [
    stacked_line.dict
    for stacked_line
    in stacked_lines
]

"""
Let's see what the first one looks like
"""
print(json.dumps(stacked_line_dicts[0], indent=4))

"""
We can save these to a JSON
"""
with open("stacked_lines.json", "w+") as f:
    json.dump(stacked_line_dicts, f)

"""
We can find the size if our CCD from the shape of the array we loaded earlier
"""
size = array.shape[0]

"""
Using this and a pixel line dictionary we can create a Dataset1D. This can
then be used with autocti.
"""
dataset_1d = Dataset1D.from_pixel_line_dict(
    stacked_line_dicts[0],
    size=size,
)

"""
The `Layout1D` object describes the layout of the data, specifically where signal which loses electrons to CTI  located
(the FPR) and where signal which gains electrons due to CTI is (the EPER). 

The `region_list` contains the pixel indexes containing charge before CTI trailing, which for warm pixels are simply
the locations of the warm pixels (e.g. the delta function values).
"""
print(dataset_1d.layout)
print(dataset_1d.layout.region_list)

"""
We can output this to .pickle files for fitting with PyAutoCTI.
"""
dataset_1d_file = Path.cwd() / "dataset" / "individual"

with open(dataset_1d_file, 'wb') as outfile:
    pickle.dump(dataset_1d, outfile)

"""
Better yet we can create a Dataset1D for every stacked pixel line we found
"""
dataset_1d_list = [
    Dataset1D.from_pixel_line_dict(
        stacked_line_dict,
        size=array.shape[0],
    )
    for stacked_line_dict
    in stacked_line_dicts
]

"""
By printing the region lists of each dataset 1d, we can see that they are grouped according to where the warm pixel is
on the data.

For example, for the first and second datasets (indexes 0 and 1) the warm pixels are located at index 13 and are very
close to the read-out register. 

For the final dataset they are at index 1629 and very far away.
"""
print(dataset_1d_list[0].layout.region_list)
print(dataset_1d_list[1].layout.region_list)
print(dataset_1d_list[-1].layout.region_list)

"""
We can output this stack of Dataset1D objects to .pickle files for fitting with PyAutoCTI.
"""
dataset_1d_file = Path.cwd() / "dataset" / "stack"

with open(dataset_1d_file, 'wb') as outfile:
    pickle.dump(dataset_1d_list, outfile)


# Ignore these currently

# """
# We can output this to .fits file for loading in PyAutoCTI
# """
# dataset_1d_output = Path.cwd() / "dataset_1d_individual"
#
# dataset_1d.output_to_fits(
#     data_path=dataset_1d_output / "data.fits",
#     noise_map_path=dataset_1d_output / "noise_map.fits",
#     pre_cti_data_path=dataset_1d_output / "pre_cti_data.fits"
# )
#
#
# """
# We can output these datasets to .fits files for fitting in another script.
# """
# dataset_1d_output = Path.cwd() / "dataset_1d_many"
#
# for i, dataset_1d in enumerate(dataset_1d_list):
#
#     dataset_1d_path = dataset_1d_output / f"dataset_1d_{i}"
#
#     dataset_1d.output_to_fits(
#         data_path=dataset_1d_path / "data.fits",
#         noise_map_path=dataset_1d_path / "noise_map.fits",
#         pre_cti_data_path=dataset_1d_path / "pre_cti_data.fits"
#     )
