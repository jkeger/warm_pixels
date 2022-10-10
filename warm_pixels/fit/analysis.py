import numpy as np
from matplotlib import pyplot as plt

import autofit as af
from autoarray.fit.fit_dataset import SimpleFit


class Analysis(af.Analysis):
    def __init__(self, pixel_line, y):
        self.pixel_line = pixel_line
        self.y = y
        self.x = np.arange(1, self.y.shape[0])

    @property
    def n_e(self):
        return self.pixel_line.n_e

    @property
    def n_bg(self):
        return self.pixel_line.n_bg

    @property
    def row(self):
        return self.pixel_line.row

    @property
    def noise(self):
        return self.pixel_line.noise

    def _call_instance(self, instance):
        return instance(
            x=self.x,
            n_e=self.n_e,
            n_bg=self.n_bg,
            row=self.row,
        )

    def visualize(self, paths, instance, during_analysis):
        plt.plot(self.x, self.y)
        plt.plot(self.x, instance(
            x=self.x,
            n_e=self.n_e,
            n_bg=self.n_bg,
            row=self.row,
        ))

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
