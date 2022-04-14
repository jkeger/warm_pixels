Warm Pixels
===========

Warm pixels and CTI trails from HST ACS, etc.

See also the paper for descriptions/explanations of the methods.

Usage
-----

The Main code file to find, stack, and plot warm pixels in multiple datasets of HST ACS images can be run from the 
command line.

```bash
./scripts/hst_warm_pixels.py data_directory \
    [--downsample] \
    [--quadrants quadrants] \
    [--plot plots] \
    [--use-corrected]
```

|Argument|Description|Example(s)|
|--------|-----------|----------|
|data_directory| Only required argument. The path to a directory containing datasets. Each dataset is a directory containing images with the same capture date.|/path/to/directory|
|--after|Filter datasets to only use images captured after a given date or number of days since launch.|123, 2022-01-01|
|--before|Filter datasets to only use images captured before a given date or number of days since launch.|123, 2022-01-01|
|--downsample|Only include every nth dataset.|3|
|--quadrants|Specify which quadrants should be included and how they should be grouped.|ABCD, AB, AB_CD|
|--plot|List which plots to output.|warm-pixels, warm-pixel-distributions, stacked-trails, density|
|--corrected|If this flag is passed the image is corrected before processing.||
