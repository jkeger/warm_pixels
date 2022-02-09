from warm_pixels import WarmPixels


def test_integration(mock_dataset):
    WarmPixels(
        datasets=[mock_dataset],
        quadrants="A",
        overwrite=True,
    ).main()
