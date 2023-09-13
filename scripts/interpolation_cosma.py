import autofit as af
import shutil
import os
from os import path
from pathlib import Path

from autofit.aggregator import search_output
from autofit.database.aggregator import scrape
from autofit.database.model import fit

import dill

search_output.pickle = dill
scrape.pickle = dill
fit.pickle = dill

# =============================================================================
# dir = os.path.join(path.sep, "mnt", "d", "L4Code", "warm_pixels", "destination")
# if not os.path.exists(dir):
#     os.mkdir(dir) 
# =============================================================================
    
# This top bit is what I used to generate data to test this on. You don't necessarily need to run it.

# =============================================================================
# class Analysis(af.Analysis):
#     def log_likelihood_function(self, instance):
#         return -0.5 * ((instance.beta + instance.rho) ** 2.0)
# 
# 
# analysis = Analysis()
# 
# 
# for t in range(10):
#     rho=t*5
#     model = af.Collection(
#         t=t,
#         beta=af.UniformPrior(lower_limit=0.0, upper_limit=10.0),
#         rho=af.UniformPrior(lower_limit=50.0, upper_limit=60.0),
#     )
# 
#     search = af.DynestyStatic(path_prefix="path_prefix",
#     name="name",
#     unique_tag='tag', number_of_cores=1)
#     search.fit(model=model, analysis=analysis)
# =============================================================================
# =============================================================================
# print("Moving output files")
# source=Path(path.join(path.sep, "mnt", "d", "L4Code", "warm_pixels", "scripts", "output", "change_dir"))
# destination=Path(path.join(path.sep, "mnt", "d", "L4Code", "warm_pixels", "destination"))
# shutil.move(source, destination)
# print("Output files moved")
# =============================================================================

# This line creates a database that is puts results in. Don't worry too much about it for now.
aggregator = af.Aggregator.from_database("database.sqlite")
# Point this line at the directory where all your results are. It should find all the results in that directory and
# load them into the aggregator.
cosma_output_path=path.join(path.sep, "cosma", "home", "durham","dc-barr6", "warm_pixels_workspace", 
                       "batch_scripts", "output", "uncorrected")
aggregator.add_directory(cosma_output_path) # Looks for zip files

# The aggregator is a way of loading all the data from multiple searches. Below is just a sanity check that the values
# are what we expect.
# =============================================================================
# for fit in sorted(aggregator, key=lambda f: f.instance.t):
#     print('instance rhos are:',fit.instance.rho)
#     print('instance betas are:',fit.instance.beta)
# =============================================================================

# We need a Samples object for each time step. Later on we can switch this out for the samples_summary.
samples = aggregator.values("samples")

# Our interpolator needs a list of samples to perform interpolation. The samples objects include the parameters
# of the best fit, the covariance matrix and information about the model. You could also
interpolator = af.CovarianceInterpolator(samples)

# We can now use the interpolator to get the value of the model at any time. This is a linear interpolation between
# the two nearest time steps.
# =============================================================================
# print('interpolated rho at t=500 days is ',interpolator[interpolator.days_var == 500].rho_q)
# print('interpolated rho at t=1550 days is ',interpolator[interpolator.days_var == 1550].rho_q)
# print('interpolated rho at t=3200 days is ',interpolator[interpolator.days_var == 3200].rho_q)
# =============================================================================
print('interpolated rho at t=6800 days is ',interpolator[interpolator.days_var == 6800].rho_q)

# We can also specify the relationship each variable has to time.
# =============================================================================
# instance = interpolator.get(
#     interpolator.t == 5.0,
#     {
#         interpolator.value: af.Model(
#             af.LinearRelationship,
#             m=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
#             c=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
#         )
#     },
# )
# print('instance rho value is:',instance.rho)
# print('instance beta value is:',instance.beta)
# =============================================================================
