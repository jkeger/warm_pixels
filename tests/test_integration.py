import numpy as np
import pytest
from autoarray.instruments import acs
from autoarray.instruments.acs import ImageACS, HeaderACS

from warm_pixels import WarmPixels, Dataset


@pytest.mark.parametrize(
    "true_flags, n_calls",
    [
        (["prep_density", "plot_density"], 7),
        (["prep_density", "plot_density", "use_corrected"], 5),
        ([], 6),
    ]
)
def test_integration(
        mock_dataset,
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
        datasets=[mock_dataset, mock_dataset],
        quadrants="A",
        overwrite=overwrite,
        **kwargs,
    ).main()
    if overwrite:
        assert len(savefig_calls) == n_calls


def test_real_dataset(
        dataset_path,
        output_path,
):
    dataset = Dataset(
        dataset_path,
        output_path,
    )
    WarmPixels(
        datasets=[dataset, dataset],
        quadrants="A",
        overwrite=True,
        prep_density=True,
        plot_density=True,
        use_corrected=True,
    ).main()
