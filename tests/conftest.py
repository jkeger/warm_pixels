import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from autoarray import acs
from autoarray.instruments.acs import ImageACS, HeaderACS
from matplotlib import pyplot

from warm_pixels.dataset import Dataset
from warm_pixels.dataset import cti
from warm_pixels.hst_functions import trail_model

directory = Path(__file__).parent


@pytest.fixture(
    name="output_path"
)
def make_output_path():
    return directory / "output"


@pytest.fixture(
    name="dataset_path"
)
def make_dataset_path():
    return directory / "dataset"


@pytest.fixture(
    name="dataset"
)
def make_dataset(
        dataset_path,
        output_path,
):
    return Dataset(
        dataset_path,
        output_path,
    )


@pytest.fixture(
    name="image"
)
def make_image(dataset_path):
    return np.load(
        str(dataset_path / "array_raw.fits")
    )


def remove_cti(
        image,
        **kwargs,
):
    return image


@pytest.fixture(
    autouse=True
)
def patch_arctic(monkeypatch):
    monkeypatch.setattr(
        cti, "remove_cti", remove_cti
    )
    monkeypatch.setattr(
        trail_model,
        "trail_model_hst_arctic",
        trail_model.trail_model_hst
    )


class SaveFig:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(
            (args, kwargs)
        )


@pytest.fixture(
    name="savefig_calls"
)
def patch_pyplot(monkeypatch):
    savefig = SaveFig()
    monkeypatch.setattr(
        pyplot,
        "savefig",
        savefig
    )
    return savefig.calls


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
    os.rename(f"{file_path}.npy", file_path)


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


@pytest.fixture(
    autouse=True
)
def remove_output(
        output_path
):
    yield
    shutil.rmtree(
        output_path,
        ignore_errors=True
    )
