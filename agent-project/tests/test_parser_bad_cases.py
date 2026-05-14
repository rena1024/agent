import pytest

from pydantic import ValidationError
from agent.parser import parse_plan


def test_missing_action():
    with pytest.raises(ValidationError):
        parse_plan({"tool": "calculator"})


def test_unknown_action():
    with pytest.raises(ValidationError):
        parse_plan({"action": "oops"})


def test_extra_field_forbid():
    with pytest.raises(ValidationError):
        parse_plan({"action": "final", "output": "x", "foo": "bar"})
