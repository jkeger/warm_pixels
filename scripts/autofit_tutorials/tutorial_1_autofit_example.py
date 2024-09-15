from matplotlib import pyplot as plt
import os.path
import pickle
from pathlib import Path

import autofit as af
from autoarray.fit.fit_dataset import SimpleFit

from warm_pixels.hst_functions.trail_model import trail_model

from warm_pixels import hst_utilities as ut

data_directory = Path(os.path.abspath("")) / "data"


def load_data(name):
    with open(data_directory / f"{name}.pickle", "r+b") as f:
        return pickle.load(f)


date = load_data("date")
x = load_data("x")
y = load_data("y")
noise = load_data("noise")
n_e = load_data("n_e")
n_bg = load_data("n_bg")
row = load_data("row")

plt.plot(x, y)

"""
We create an analysis class to hold data and compute how good a fit each instance
is. The instance is some callable that given x, n_e, n_bg and row will return a
trail modelled using a set of parameters sampled from parameter space.

The analysis can also use this functionality to visualise the modelled trail and
show how good a match it is.
"""


class Analysis(af.Analysis):
    def __init__(self, x, y, noise, n_e, n_bg, row):
        self.x = x
        self.y = y
        self.noise = noise
        self.n_e = n_e
        self.n_bg = n_bg
        self.row = row

    def visualize(self, paths, instance, during_analysis):
        plt.plot(self.x, self.y)
        plt.plot(
            self.x,
            instance(
                x=self.x,
                n_e=self.n_e,
                n_bg=self.n_bg,
                row=self.row,
            ),
        )

    def log_likelihood_function(self, instance):
        modelled_trail = instance(
            x=self.x,
            n_e=self.n_e,
            n_bg=self.n_bg,
            row=self.row,
        )
        fit = SimpleFit(
            data=self.y,
            model_data=modelled_trail,
            noise_map=self.noise,
        )
        return fit.log_likelihood


"""
The TrailModel is what we're fitting as we optimise. It's parameterised by the arguments
passed into the constructor (rho_q, beta, w, a, b, c, tau_a, tau_b and tau_c). These
arguments can be constant or they can be a prior if we want the variable to be optimised.

The trail model can be called to recover a modelled trail.
"""


class TrailModel:
    def __init__(
        self,
        rho_q,
        beta,
        w,
        a,
        b,
        c,
        tau_a,
        tau_b,
        tau_c,
    ):
        self.rho_q = rho_q
        self.beta = beta
        self.w = w
        self.a = a
        self.b = b
        self.c = c
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.tau_c = tau_c

    def __call__(self, x, n_e, n_bg, row):
        return trail_model(
            x=x,
            rho_q=self.rho_q,
            n_e=n_e,
            n_bg=n_bg,
            row=row,
            beta=self.beta,
            w=self.w,
            A=self.a,
            B=self.b,
            C=self.c,
            tau_a=self.tau_a,
            tau_b=self.tau_b,
            tau_c=self.tau_c,
        )


"""
For now we're setting constant values for most parameters
"""
# CCD
beta = 0.478
w = 84700.0
# Trap species
a = 0.17
b = 0.45
c = 0.38
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
We make a model using the TrailModel type and passing kwargs for each of the constructor arguments.

rho_q is a variable. It can take any value between 0 and 10 with a uniform probability. All the
other arguments are constants.
"""
model = af.Model(
    TrailModel,
    rho_q=af.UniformPrior(
        lower_limit=0.0,
        upper_limit=10.0,
    ),
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
We make an instance of the analysis, passing it the data and associated variables.
"""
analysis = Analysis(
    x=x,
    y=y,
    noise=noise,
    n_e=n_e,
    n_bg=n_bg,
    row=row,
)

"""
We make our optimiser which we can configure
"""
dynesty = af.DynestyStatic()

"""
Optimisation is performed by passing the model and the analysis to the fit function
"""
result = dynesty.fit(
    model=model,
    analysis=analysis,
)

"""
The log likelihood and best parametrisation can be found from the result
"""
print(f"log likelihood = {result.log_likelihood}")

best_trail_model = result.instance

print(f"rho_q = {best_trail_model.rho_q}")
