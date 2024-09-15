from pathlib import Path

import warm_pixels as wp

"""
We assume our data is in a directory parallel to warm_pixels called 'hst_acs_datasets'
"""
data_directory = Path.cwd().parent / "hst_acs_datasets"

"""
We can create an object to access all datasets in that directory
"""
source = wp.Source(data_directory)
print(f"Source contains {len(source)} datasets")

first = source[0]
last = source[-1]
print(f"First dataset {first} ({first.days_since_launch()})")
print(f"Last dataset {last} ({last.days_since_launch()})")

"""
Let's filter out a dataset from before a given date
"""
source = source.after(1629)
first = source[0]
print(f"Source contains {len(source)} datasets")
print(f"First dataset {first} ({first.days_since_launch()})")

"""
We can downsample too. We skip to every nth dataset.
"""
source = source.downsample(2)
print(f"Source contains {len(source)} datasets")

"""
A warm pixels object can easily be created from our source
"""
warm_pixels = wp.WarmPixels(datasets=list(source), quadrants_string="AB")

"""
And from this we can create a plot object
"""
plot = wp.plot.Plot(
    warm_pixels,
    list_name=str(source),
)

"""
We can use this object to produce plots
"""
plot.density()

"""
Note that the filenames for plots are generated to be unique. Files are output to the output/ directory.

It's also possible to do the same but for corrected data
"""
source = source.corrected()
warm_pixels = warm_pixels(datasets=list(source), quadrants_string="AB")

plot = wp.plot.Plot(
    warm_pixels,
    list_name=str(source),
    use_corrected=True,
)

plot.warm_pixels()
