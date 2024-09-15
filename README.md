Warm Pixels
===========

Warm pixels and CTI trails from HST ACS, etc.

See also the paper for descriptions/explanations of the methods.


Usage
-----

The main code file to find, stack, and plot warm pixels in multiple datasets of
HST ACS images can be run from the command line:

```bash
./scripts/hst_warm_pixels.py data_directory \
    [--downsample] \
    [--quadrants quadrants] \
    [--plot plots] \
    [--ouput outputs] \
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
|--output|List which outputs to save.|consistent_lines, stacked_lines|
|--corrected|If this flag is passed the image is corrected before processing.||



\
Files
-----

+ `scripts/`
    + `hst_warm_pixels.py`  
        Main script to find, stack, and plot warm pixels in datasets of HST images.
    + `paper_plots.py`  
        Alternative main script to make tweaked plots and examples for the paper.
    + `tutorial_*.py`  
        Tutorials.
    + `autofit_tutorials/tutorial_*.py`  
        Tutorials for autofit.
+ `warm_pixels/`
    + `warm_pixels.py`  
        Primary function to identify warm pixels in an image.
    + `hst_utilities.py`  
        Utility functions and input values, mainly for HST-specific use.
    + `bins.py`  
        The Bins class for organising data.
    + `func_test.py`  
        The FuncTest class with test utilities.
    + `misc.py`  
        Miscellaneous plotting-etc utilities and defaults.
    + `output.py`  
        Generic and json output utilities.
    + `pixel_lines/`
        + `pixel_line.py`  
            The PixelLine class and utilities for managing 1D lines of pixel data.
        + `collection.py`  
            Classes and utilities for collections of pixel lines, including
            finding consistent lines across a set and stacking lines in bins.
        + `stacked_collection.py`  
            Classes and utilities for stacked pixel lines and collections of them.
    + `hst_functions/`
        + `trail_model.py`  
            Functions to model trails with exponentials or ArCTIc, and specific HST versions.
        + `cti_model.py`  
            Set preset HST-model objects for ArCTIc.
        + `fit.py`  
            Functions to run the model-parameter fitting over image datasets.
    + `fit/`
        + `analysis.py`  
            Wrapper class for autofit analysis utilities.
        + `model.py`  
            Wrapper class for a sum-of-exponentials model trail.
    + `model/`
        + `quadrant.py`  
            Classes to manage image quadrants and datasets of them.
        + `group.py`  
            Class to manage groups of quadrants.
        + `cache.py`  
            Utilities for caching the output of functions to avoid repeating slow steps.
    + `plot/`
        + `warm_pixels.py`  
            Functions to plot found locations and histograms of found warm pixels.
        + `stacked_trails.py`  
            Function to plot a tiled set of stacked trails.
        + `trap_density.py`  
            Function to plot the evolution of total HST trap density.
    + `data/`
        + `image.py`  
            Class to load HST images.
        + `dataset.py`  
            Classes to manage datasets of HST images.
        + `source.py`  
            Classes for managing datasets with different origin image sources.
+ `notebooks/*.ipynb`  
    Tutorial notebooks.
    + `autofit/*.ipynb`  
        Tutorial notebooks for autofit.
+ `tests/`
    + `test_*.py`  
        Unit tests, run with `pytest tests/`.
    + `conftest.py`  
        Modules and fixtures for the unit tests.
    + `fit_total_trap_density`  
        ##?
    + `combined_lines.pickle`  
        ##?
    + `dataset_list/dataset/array_raw.fits`  
        ##?
