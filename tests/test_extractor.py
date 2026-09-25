import pytest

from server.models import EventType
from server.nlp_extractor import ATCExtractor


@pytest.fixture(scope="module")
def ex():
    return ATCExtractor()


def canon(ex, text):
    return [c["canonical"] for c in ex.extract_all(text, 0.0)["callsigns"]]


@pytest.mark.parametrize("text,expected", [
    ("Air Canada eight eight five three, contact tower", ["ACA8853"]),
    ("Delta 1492 heavy runway 28 left cleared to land", ["DAL1492"]),
    ("united fourteen ninety two climb", ["UAL1492"]),
    ("Jazz 8-8-5-3 hold short", ["JZA8853"]),
    ("November one two three alpha bravo turn right", ["N123AB"]),
    ("Charlie Golf Alpha Bravo Charlie, join downwind", ["C-GABC"]),
    ("Foxtrot Kilo Lima Mike, taxi to holding point", ["C-FKLM"]),
    ("ACA8853 and WJA123 on frequency", ["ACA8853", "WJA123"]),
    ("Speedbird 27A descend", ["BAW27A"]),
])
def test_callsigns(ex, text, expected):
    assert canon(ex, text) == expected


def test_flight_level_is_not_a_callsign(ex):
    assert canon(ex, "climb FL240") == []


def test_alias_is_substring_of_normalized_text(ex):
    out = ex.extract_all("delta one four nine two descend", 0.0)
    assert out["normalized"] == "delta 1492 descend"
    assert out["callsigns"][0]["alias"] == "delta 1492"


def fields(ex, text):
    return ex.extract_all(text, 0.0)["fields"]


def test_digit_output_fields(ex):
    # Stock Whisper writes digits; the old extractor only read the first one.
    f = fields(ex, "Delta 1492 turn left heading 270 descend and maintain 3000 "
                   "reduce speed to 180 contact approach 124.6")
    assert f.heading == "270°"
    assert f.altitude == "3000ft"
    assert f.speed == "180kts"
    assert f.frequency == "124.6"


def test_spoken_fields(ex):
    f = fields(ex, "jazz eight eight five three climb flight level two four zero "
                   "squawk four five two one runway two four right")
    assert f.altitude == "FL240"
    assert f.squawk == "4521"
    assert f.runway == "24R"


def test_altitude_thousands(ex):
    assert fields(ex, "descend and maintain three thousand five hundred").altitude == "3500ft"


def test_bare_maintain_speed_not_altitude(ex):
    f = fields(ex, "maintain 250 knots")
    assert f.altitude is None
    assert f.speed == "250kts"


def test_frequency_spaced(ex):
    assert fields(ex, "contact Minneapolis center 124 85").frequency == "124.85"


def test_runway_padding(ex):
    assert fields(ex, "cleared to land runway 6 left").runway == "06L"


def events(ex, text):
    return [e.type for e in ex.extract_all(text, 0.0)["events"]]


def test_hold_short_is_not_generic_hold(ex):
    assert events(ex, "hold short runway 24R") == [EventType.HOLD_SHORT]


def test_hold_position_is_not_an_event(ex):
    assert events(ex, "hold position") == []


def test_alternate_missed_is_not_divert(ex):
    assert EventType.DIVERT not in events(ex, "alternate missed approach instructions")


def test_mayday(ex):
    assert events(ex, "mayday mayday mayday") == [EventType.MAYDAY]


def test_squawk_7700(ex):
    assert EventType.EMERGENCY in events(ex, "squawk seven seven zero zero")


def test_go_around_links_callsign(ex):
    out = ex.extract_all("Delta 1492 go around", 1.0)
    assert out["events"][0].type == EventType.GO_AROUND
    assert out["events"][0].callsign_canonical == "DAL1492"
