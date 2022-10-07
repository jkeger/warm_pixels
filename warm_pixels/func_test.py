import pickle

import pytest


class FuncTest:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)

        with open(self.func.__name__, "w+b") as f:
            pickle.dump((args, kwargs, result), f)

        return result

    def test(self):
        with open(self.func.__name__, "r+b") as f:
            args, kwargs, result = pickle.load(f)

        # for key, value in zip(
        #         ("x_all", "y_all", "noise_all", "n_e_all", "n_bg_all", "row_all", "date"), args
        # ):
        #     with open(f"{key}.pickle", "w+b") as f:
        #         pickle.dump(value, f)

        new_result = self.func(*args, **kwargs)
        assert new_result[0] == pytest.approx(result[0], rel=0.1)
        assert new_result[1] == pytest.approx(result[1], rel=0.1)