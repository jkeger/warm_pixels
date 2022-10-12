"""
Modeling: Dataset 1D x2 Species
===============================

In this script, we will fit a 1D CTI Dataset to calibrate a CTI model, where:

 - The CTI model consists of multiple parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
from pathlib import Path
import pickle
import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Dataset__

Load the cti dataset 'dataset_1d/species_x2' 'from .fits files, which is the dataset we will use to perform CTI modeling.
"""
dataset_1d_file = Path.cwd() / "dataset" / "stack"

"""
__Dataset__

We now load every cti-dataset, including a noise-map and pre-cti data containing the data before read-out and
therefore without CTI. This uses a`Dataset1D` object.
"""
with open(dataset_1d_file, 'rb') as outfile:
    dataset_1d_list = pickle.load(outfile)

"""
__Masking__

We now mask every 1D dataset, removing the FPR of each dataset so we use only the EPER to calibrate the CTI model.
"""
mask = ac.Mask1D.unmasked(
    shape_slim=dataset_1d_list[0].shape_slim,
    pixel_scales=dataset_1d_list[0].pixel_scales,
)

mask = ac.Mask1D.masked_fprs_and_epers_from(
    mask=mask,
    layout=dataset_1d_list[0].layout,
    settings=ac.SettingsMask1D(fpr_pixels=(0, 10)),
    pixel_scales=dataset_1d_list[0].pixel_scales,
)

dataset_1d_list = [dataset_1d.apply_mask(mask=mask) for dataset_1d in dataset_1d_list]

"""
Lets plot the first dataset.
"""
dataset_1d_plotter = aplt.Dataset1DPlotter(dataset=dataset_1d_list[0])
dataset_1d_plotter.subplot_dataset_1d()

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 
"""
clocker_1d = ac.Clocker1D(express=2)

"""
__Model__

We now compose our CTI model, which represents the trap species and CCD volume filling behaviour used to fit the cti 
1D data. In this example we fit a CTI model with:

 - Two `TrapInstantCapture`'s which capture electrons during clocking instantly in the parallel direction
 [4 parameters].
 
 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.
"""
trap_0 = af.Model(ac.TrapInstantCapture)
trap_1 = af.Model(ac.TrapInstantCapture)

trap_0.add_assertion(trap_0.release_timescale < trap_1.release_timescale)

trap_list = [trap_0, trap_1]

ccd = af.Model(ac.CCDPhase)
ccd.well_notch_depth = 0.0
ccd.full_well_depth = 200000.0

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Search__

The CTI model is fitted to the data using a `NonLinearSearch`. In this example, we use the
nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/).

The script 'autocti_workspace/examples/modeling/customize/non_linear_searches.py' gives a description of the types of
non-linear searches that can be used with **PyAutoCTI**. If you do not know what a `NonLinearSearch` is or how it 
operates, checkout chapter 2 of the HowToCTI lecture series.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autocti_workspace/output/dataset_1d/species[x2]`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("warm_pixels"), name="species[x2]", nlive=50
)

"""
__Analysis__

The `AnalysisDataset1D` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
to the `Dataset1D`dataset.
"""
analysis_list = [
    ac.AnalysisDataset1D(dataset=dataset_1d, clocker=clocker_1d)
    for dataset_1d in dataset_1d_list
]

"""
By summing this list of analysis objects, we create an overall `Analysis` which we can use to fit the CTI model, where:

 - The log likelihood function of this summed analysis class is the sum of the log likelihood functions of each 
 individual analysis object.

 - The summing process ensures that takes such as outputting results to hard-disk, visualization, etc use a 
 structure that separates each analysis.
"""
analysis = sum(analysis_list)

"""
We can parallelize the likelihood function of these analyis classes, whereby each evaluation will be performd on a 
different CPU.
"""
analysis.n_cores = 1

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autocti_workspace/output/dataset_1d/species[x2]` for live outputs of the results of the fit, 
including on-the-fly visualization of the best fit model!
"""
result_list = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by the fit provides information on the results of the non-linear search. 

The `info` attribute shows the result in a readable format.
"""
print(result_list.info)

"""
The result object also contains the fit corresponding to the maximum log likelihood solution in parameter space,
which can be used to visualizing the results.
"""
print(result_list[0].max_log_likelihood_instance.cti.trap_list[0].density)
print(result_list[0].max_log_likelihood_instance.cti.ccd.well_fill_power)

for result in result_list:

    fit_plotter = aplt.FitDataset1DPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit_dataset_1d()

"""
Checkout `autocti_workspace/*/dataset_1d/modeling/results.py` for a full description of the result object.
"""
