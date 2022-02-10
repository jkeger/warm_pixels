from warm_pixels.warm_pixels import find_warm_pixels, find_dataset_warm_pixels


def test_warm_pixels(
        image
):
    result = find_warm_pixels(
        image=image,
        trail_length=2,
        ignore_bad_columns=False,
    )
    assert len(result) == 107


def test_dataset_warm_pixels(
        dataset_path,
        image,
        mock_dataset
):
    result = find_dataset_warm_pixels(
        mock_dataset,
        quadrant="A"
    )

    assert len(result) == 43
