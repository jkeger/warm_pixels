import numpy as np

from warm_pixels import PixelLine


def test_to_dict():
    pixel_line = PixelLine(
        location=[26, 25],
        date=20.5,
        background=23,
        flux=150,
        data=np.array([1.0, 4.0, 5.0]),
        noise=np.array([2.0, 3.0, 4.0])
    )

    assert pixel_line.dict == {
        'background': 23,
        'data': [1., 4., 5.],
        'date': 20.5,
        'flux': 150,
        'location': [26, 25],
        'noise': [2., 3., 4.]
    }
