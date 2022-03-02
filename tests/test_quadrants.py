import pytest

from warm_pixels.quadrant_groups import QuadrantsString


@pytest.mark.parametrize(
    "string, groups",
    [
        ("ABCD", (("A", "B", "C", "D",),)),
        ("AB_CD", (("A", "B",), ("C", "D",),)),
        ("A_B_C_D", (("A",), ("B",), ("C",), ("D",),)),
    ]
)
def test_quadrant_groups(
        string, groups
):
    quadrants = QuadrantsString(string)
    assert quadrants.groups == groups


@pytest.mark.parametrize(
    "string",
    [
        "ABCD",
        "AB_CD",
        "A_B_C_D",
    ]
)
def test_iter(string):
    quadrants = QuadrantsString(string)

    assert list(quadrants) == [
        "A", "B", "C", "D"
    ]


def test_len():
    assert len(QuadrantsString("ABCD")) == 4
