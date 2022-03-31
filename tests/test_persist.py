from warm_pixels.model.quadrant import ImageQuadrant


def test_persist_warm_pixels(
        image,
        output_path
):
    quadrant = ImageQuadrant(
        image=image,
        quadrant="A"
    )
    quadrant.warm_pixels()

    assert (output_path / quadrant.name / "warm_pixels.pickle").exists()
