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


def output_quadrants_to_fits(
        file_path: str,
        quadrant_a,
        quadrant_b,
        quadrant_c,
        quadrant_d,
        header_a=None,
        header_b=None,
        header_c=None,
        header_d=None,
        overwrite: bool = False,
):
    np.save(
        file_path,
        quadrant_a
    )


def from_fits(
        file_path,
        quadrant_letter,
        bias_subtract_via_bias_file=False,
        bias_subtract_via_prescan=False,
        bias_file_path=None,
        use_calibrated_gain=True,
):
    image = np.load(file_path)
    return ImageACS(
        np.load(file_path),
        image.shape,
        header=HeaderACS(
            header_sci_obj={
                "DATE-OBS": "2020-01-01",
                "TIME-OBS": "15:00:00",
                "CCDGAIN": 1,
            },
            header_hdu_obj={
                "BSCALE": 1,
                "BZERO": 1,
                "BUNIT": "COUNTS",
            },
            hdu=None,
            quadrant_letter=quadrant_letter,
        )
    )


@pytest.fixture(
    autouse=True
)
def patch_fits(monkeypatch):
    monkeypatch.setattr(
        ImageACS,
        "from_fits",
        from_fits
    )
    monkeypatch.setattr(
        acs,
        "output_quadrants_to_fits",
        output_quadrants_to_fits
    )
