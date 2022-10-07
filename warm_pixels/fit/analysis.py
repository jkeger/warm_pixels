import autofit as af

from autoarray.fit.fit_dataset import SimpleFit


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
