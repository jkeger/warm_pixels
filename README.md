Warm Pixels Etc
===============

Warm pixels and CTI trails from HST ACS, etc.

This code is hopefully documented well enough for others to understand and work
on, but since it's intended just for our own use rather than as a public tool
I'm sure there are plenty of bits that aren't completely clear or immediately
intuitive, not least with the over-engineered sys arg setup I was playing with.
It should at least be divided into fairly clean functions and steps.

See also the paper for descriptions/explanations of the methods.

Contents
--------
+ `hst_warm_pixels.py`
    Main code file to find, stack, and plot warm pixels in multiple datasets of
    HST ACS images. See the docstring for the details and pipeline, and the
    Datasets class and lists of dataset names for the sets of HST images, which
    you might want to modify.

    Run with the different flag arguments to select which functions to run,
    many of which save their output data to be loaded and used in the next step.
    Most will run in order automatically to analyse the images in the selected
    datasets, but any can be enabled or disabled. Might look a bit weird at
    first but it made it fairly convenient for running and re-running different
    steps while not repeating steps that have already been done.
+ `hst_functions.py`
    The main function contents, including making plots.
+ `hst_utilities.py`
    Simple utility functions and core inputs like directory paths and warm-pixel
    parameters.
+ `paper_plots.py`
    Plotting functions for the paper. Most are fairly simple and/or ~duplicates
    of main functions with aesthetic tweaks.
+ `pixel_lines.py`, `warm_pixels.py`, `test_pixel_lines.py`, `test_warm_pixels.py`
    Core library modules, see their docstrings, and unit tests.
