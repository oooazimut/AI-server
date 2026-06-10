from datetime import timedelta

import pytest

from ai_server.utils import MOSCOW_TZ, compact_text, confidence, optional_int, truthy, unique


# optional_int
def test_optional_int_none():
    assert optional_int(None) is None


def test_optional_int_empty_string():
    assert optional_int("") is None


def test_optional_int_valid_string():
    assert optional_int("42") == 42


def test_optional_int_integer():
    assert optional_int(7) == 7


def test_optional_int_float_truncates():
    assert optional_int(3.9) == 3


def test_optional_int_invalid_string():
    assert optional_int("abc") is None


def test_optional_int_list():
    assert optional_int([]) is None


def test_optional_int_negative():
    assert optional_int("-5") == -5


# unique
def test_unique_preserves_order():
    assert unique(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_unique_empty():
    assert unique([]) == []


def test_unique_all_same():
    assert unique(["x", "x", "x"]) == ["x"]


def test_unique_no_duplicates():
    assert unique(["a", "b", "c"]) == ["a", "b", "c"]


def test_unique_single_element():
    assert unique(["only"]) == ["only"]


# confidence
def test_confidence_zero():
    assert confidence(0.0) == 0.0


def test_confidence_one():
    assert confidence(1.0) == 1.0


def test_confidence_clamps_below_zero():
    assert confidence(-1.0) == 0.0


def test_confidence_clamps_above_one():
    assert confidence(2.0) == 1.0


def test_confidence_string_float():
    assert confidence("0.7") == pytest.approx(0.7)


def test_confidence_invalid_string_returns_half():
    assert confidence("bad") == 0.5


def test_confidence_none_returns_half():
    assert confidence(None) == 0.5


def test_confidence_midpoint():
    assert confidence(0.5) == pytest.approx(0.5)


# truthy
def test_truthy_bool_true():
    assert truthy(True) is True


def test_truthy_bool_false():
    assert truthy(False) is False


def test_truthy_int_one():
    assert truthy(1) is True


def test_truthy_int_zero():
    assert truthy(0) is False


def test_truthy_float_nonzero():
    assert truthy(0.1) is True


def test_truthy_string_da():
    assert truthy("да") is True


def test_truthy_string_yes():
    assert truthy("yes") is True


def test_truthy_string_one():
    assert truthy("1") is True


def test_truthy_string_true():
    assert truthy("true") is True


def test_truthy_string_y():
    assert truthy("y") is True


def test_truthy_string_bez_sroka():
    assert truthy("без срока") is True


def test_truthy_string_bessrochno():
    assert truthy("бессрочно") is True


def test_truthy_string_false():
    assert truthy("false") is False


def test_truthy_string_no():
    assert truthy("no") is False


def test_truthy_string_with_whitespace():
    assert truthy("  true  ") is True


def test_truthy_list_nonempty():
    assert truthy(["a"]) is True


def test_truthy_list_empty():
    assert truthy([]) is False


# compact_text
def test_compact_text_collapses_spaces():
    assert compact_text("  a  b  ") == "a b"


def test_compact_text_collapses_newlines():
    assert compact_text("a\n\nb") == "a b"


def test_compact_text_collapses_tabs():
    assert compact_text("a\t\tb") == "a b"


def test_compact_text_already_compact():
    assert compact_text("hello world") == "hello world"


def test_compact_text_empty():
    assert compact_text("") == ""


def test_compact_text_only_whitespace():
    assert compact_text("   ") == ""


# MOSCOW_TZ
def test_moscow_tz_offset():
    assert MOSCOW_TZ.utcoffset(None) == timedelta(hours=3)
