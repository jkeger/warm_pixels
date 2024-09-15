from pathlib import Path

import warm_pixels as wp

"""
We assume our data is in a directory parallel to warm_pixels called 'hst_acs_datasets'
"""
data_directory = Path.cwd().parent / "hst_acs_datasets"
dataset_directory = data_directory / "04_2017"
image_filename = dataset_directory / "jd5h08qoq_raw.fits"
wp_output = Path.cwd() / "plots"

"""
An image can be created by passing a filename
"""
image = wp.Image(image_filename)

"""
We can easily extract a particular quadrant from the image
"""
image_quadrant = image.quadrant("A")

"""
And plot warm pixels for that image
"""
wp.plot.warm_pixels(image_quadrant)

"""
A dataset can be created by passing directory
"""
dataset = wp.Dataset(dataset_directory)

"""
Datasets contain images
"""
print(len(dataset))
image = dataset[0]
wp.plot.warm_pixels(image.quadrant("B"), save_path=wp_output / "warm_pixels_plot.png")

"""
We can also take a quadrant for a dataset
"""
dataset_quadrants = [
    dataset.quadrant("A"),
    dataset.quadrant("B"),
]

"""
And then plot the distribution of warm pixels
"""
wp.plot.warm_pixel_distributions(
    dataset_quadrants, save_path=wp_output / "warm_pixel_distribution.png"
)

"""
Groups of quadrants can be obtained for a dataset
"""
group = dataset.group("AB")

"""
Stacked trails may be plotted for a group
"""
wp.plot.stacked_trails(group, save_path=wp_output / "stacked_trails.png")

"""
A WarmPixels object can be created to encapsulate multiple datasets
"""

dataset_2 = wp.Dataset(data_directory / "10_2017")

warm_pixels = wp.WarmPixels(
    datasets=[
        dataset,
        dataset_2,
    ],
    quadrants_string="A_B",
)

"""
Warm pixels can provide trap densities over time
"""
trap_densities = warm_pixels.all_trap_densities()

"""
Which can also be plotted
"""
wp.plot.trap_density_evol(trap_densities, save_path=wp_output / "trap_density_evol.png")
