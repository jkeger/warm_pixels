from warm_pixels.model.quadrant import Quadrant
from warm_pixels.warm_pixels import find_warm_pixels


def test_warm_pixels(
        array
):
    result = find_warm_pixels(
        image=array,
        trail_length=2,
        ignore_bad_columns=False,
    )
    assert len(result) == 107


def test_dataset_warm_pixels(
        dataset
):
    dataset_quadrant = Quadrant(
        quadrant="A",
        dataset=dataset,
    )
    result = dataset_quadrant.warm_pixels()

    assert len(result) == 43
