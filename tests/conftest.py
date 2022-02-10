from pathlib import Path

import numpy as np
import pytest
from autoarray import acs
from autoarray.instruments.acs import ImageACS, HeaderACS
from matplotlib import pyplot

from warm_pixels.hst_data import Dataset, Image
from warm_pixels.hst_functions import trail_model
from warm_pixels.hst_functions.cti_model import cti

directory = Path(__file__).parent
output_path = directory / "output"


@pytest.fixture(
    name="dataset_path"
)
def make_dataset_path():
    return directory / "dataset"


class MockImage(Image):
    name = "image"

    def __init__(self, array, dataset_path):
        self.array = array
        super().__init__(
            path=dataset_path / self.name,
            output_path=output_path,
        )

    def load_quadrant(
            self,
            quadrant
    ):
        return self.array

    def date(self):
        return 2400000.5 + 59049.90211805556

    def corrected(self):
        return self


class MockDataset(Dataset):
    # noinspection PyMissingConstructor
    def __init__(
            self,
            images,
            path
    ):
        self.images = images
        self.path = path

    @property
    def _output_path(self):
        return self.path


@pytest.fixture(
    name="image"
)
def make_image(dataset_path):
    return np.load(
        str(dataset_path / "array.npy")
    )


@pytest.fixture(
    name="mock_dataset"
)
def make_mock_dataset(image, dataset_path):
    return MockDataset(
        images=[MockImage(
            ImageACS(
                image,
                np.zeros(image.shape),
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
                    quadrant_letter="A",
                )
            ),
            dataset_path=dataset_path
        )],
        path=dataset_path
    )


def remove_cti(
        image,
        **kwargs,
):
    return image


def output_quadrants_to_fits(*args, **kwargs):
    pass


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


@pytest.fixture(
    autouse=True
)
def patch_auto_array(monkeypatch):
    monkeypatch.setattr(
        acs,
        "output_quadrants_to_fits",
        output_quadrants_to_fits
    )


class SaveFig:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(
            (args, kwargs)
        )


@pytest.fixture(
    autouse=True,
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
