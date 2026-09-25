import pytest

from server.text_normalizer import normalize_numbers


@pytest.mark.parametrize("text,expected", [
    ("delta one four nine two descend and maintain three thousand five hundred",
     "delta 1492 descend and maintain 3500"),
    ("United fourteen ninety-two, runway two eight left", "United 1492, runway 28 left"),
    ("contact departure one two four point eight five", "contact departure 124.85"),
    ("climb flight level three five zero", "climb flight level 350"),
    ("Jazz 8-8-5-3 maintain 3,500", "Jazz 8853 maintain 3500"),
    ("eight 8 five 3", "8853"),
    ("squawk seven seven zero zero", "squawk 7700"),
    ("one one thousand", "11000"),
    ("one hundred twenty knots", "120 knots"),
    ("heading two seven zero", "heading 270"),
    ("Delta 14 92", "Delta 1492"),
    ("tower one one eight decimal seven", "tower 118.7"),
    ("niner tree fife", "935"),
])
def test_normalize(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize("text", [
    "one moment please",
    "Oh, roger",
    "a thousand feet",
    "cleared for takeoff",
    "",
])
def test_leaves_non_numbers_alone(text):
    assert normalize_numbers(text) == text
