import json
from matplotlib import pyplot as plt
import os
from collections import defaultdict
from pathlib import Path

from warm_pixels.fit.analysis import Analysis
from warm_pixels.fit.model import TrailModel
from warm_pixels import hst_utilities as ut, PixelLine
import autofit as af

data_directory = Path(os.path.abspath("")) / "data"

"""
./hst_warm_pixels.py ../hst_acs_datasets --output stacked-lines
"""
with open(data_directory / "stacked_lines.json") as f:
    stacked_lines = json.load(f)

date_groups = defaultdict(list)

for stacked_line in stacked_lines:
    date_groups[stacked_line['date']].append(stacked_line)

print({
    date: len(lines)
    for date, lines
    in date_groups.items()
})

"""
We define global parameters which are true for any image captured
"""
# CCD
w = 84700.0
# Trap species
a = 0.17
b = 0.45
c = 0.38

"""
The global beta is allowed to vary
"""
beta = af.GaussianPrior(
    mean=0.478,
    sigma=0.1,
)

analyses = []

"""
For each date we'll need a different model
"""
for date, pixel_lines in date_groups.items():

    """
    First we set constant values for the trap lifetimes. These could also
    be allowed to vary
    """
    # Trap lifetimes before or after the temperature change
    if date < ut.date_T_change:
        tau_a = 0.48
        tau_b = 4.86
        tau_c = 20.6
    else:
        tau_a = 0.74
        tau_b = 7.70
        tau_c = 37.0

    """
    For each date we have a variable rho_q
    """
    rho_q = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=10.0,
    )

    """
    Each date has its own model. Note that the beta prior is shared
    by every model, whilst each model has its own rho_q
    """
    model = af.Model(
        TrailModel,
        rho_q=rho_q,
        beta=beta,
        w=w,
        a=a,
        b=b,
        c=c,
        tau_a=tau_a,
        tau_b=tau_b,
        tau_c=tau_c,
    )

    """
    For each pixel line we need a unique analysis. Each of these analyses
    is combined with the model for the date on which the pixel line was
    captured.
    """
    for pixel_line in pixel_lines:
        analysis = Analysis(
            pixel_line=PixelLine.from_dict(pixel_line),
        ).with_model(model)
        analyses.append(analysis)

"""
Finally we create our overall analysis by summing the other analyses together.
"""
analysis = sum(analyses)

"""
We make our optimiser which we can configure
"""
dynesty = af.DynestyStatic()

"""
Optimisation is performed by passing the model and the analysis to the fit function
"""
result = dynesty.fit(
    model=None,
    analysis=analysis,
)

"""
There are 10 instances in results which is one for every pixel line.

The beta value for each instance is the same whilst the rho_q is only
the same for instances with the same date.
"""
for instance in result.instance:
    print(f"rho_q = {instance.rho_q}")
    print(f"beta = {instance.beta}")

"""
Let's plot this to see what happens over time.
"""
date_rho_dict = {
    analysis.pixel_line.date: instance.rho_q
    for instance, analysis
    in zip(result.instance, analyses)
}

dates, rho_qs = zip(*sorted(date_rho_dict.items(), key=lambda t: t[0]))

plt.plot(dates, rho_qs)
