import pytest

from warm_pixels import WarmPixels
from warm_pixels.quadrant_groups import Quadrants


@pytest.mark.parametrize(
    "true_flags, n_calls",
    [
        # (["prep_density", "plot_density"], 7),
        # (["prep_density", "plot_density", "use_corrected"], 5),
        # ([], 6),
        (["prep_density", "plot_density"], 25),
        (["prep_density", "plot_density", "use_corrected"], 7),
        ([], 22),
    ]
)
def test_integration(
        dataset,
        true_flags,
        savefig_calls,
        n_calls,
):
    kwargs = {
        flag: True
        for flag
        in true_flags
    }
    overwrite = True
    WarmPixels(
        datasets=[dataset, dataset],
        quadrants=Quadrants("A"),
        overwrite=overwrite,
        **kwargs,
    ).main()
    if overwrite:
        assert len(savefig_calls) == n_calls
