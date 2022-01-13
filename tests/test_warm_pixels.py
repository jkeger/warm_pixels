import numpy as np
from autoarray.instruments.acs import ImageACS, HeaderACS

from warm_pixels.hst_data import Dataset
from warm_pixels.hst_functions import find_dataset_warm_pixels


class MockImage:
    name = "image"

    def __init__(self, array):
        self.array = array

    def load_quadrant(
            self,
            quadrant
    ):
        return self.array


class MockDataset(Dataset):
    # noinspection PyMissingConstructor
    def __init__(
            self,
            images,
            path
    ):
        self.images = images
        self.path = path


def test_dataset_warm_pixels(
        dataset_path
):
    result = find_dataset_warm_pixels(
        MockDataset(
            images=[MockImage(
                ImageACS(
                    np.zeros((100, 100)),
                    np.zeros((100, 100)),
                    header=HeaderACS(
                        header_sci_obj={
                            "DATE-OBS": "2020-01-01",
                            "TIME-OBS": "15:00:00",
                        },
                        header_hdu_obj=None,
                        hdu=None,
                        quadrant_letter="A",
                    )
                )
            )],
            path=dataset_path
        ),
        quadrant="A"
    )

    assert len(result) == 0
