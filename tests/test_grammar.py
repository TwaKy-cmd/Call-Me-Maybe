"""Tests for src.grammar: read-only lookups over function definitions."""

import pytest

from src.grammar import Grammar
from src.models import Function, Parameter


@pytest.fixture
def grammar() -> Grammar:
    functions = [
        Function(
            name="fn_add",
            description="Add two numbers.",
            parameters={"a": Parameter(type="number"), "b": Parameter(type="number")},
            returns=Parameter(type="number"),
        ),
        Function(
            name="fn_greet",
            description="Greet someone.",
            parameters={"name": Parameter(type="string")},
            returns=Parameter(type="string"),
        ),
    ]
    return Grammar(functions=functions)


def test_get_function_names(grammar: Grammar) -> None:
    assert grammar.get_function_names() == ["fn_add", "fn_greet"]


def test_get_function_by_name(grammar: Grammar) -> None:
    function = grammar.get_function_by_name("fn_greet")
    assert function.description == "Greet someone."


def test_get_function_by_name_missing_raises(grammar: Grammar) -> None:
    with pytest.raises(ValueError):
        grammar.get_function_by_name("does_not_exist")


def test_get_parameter_names(grammar: Grammar) -> None:
    assert grammar.get_parameter_names("fn_add") == ["a", "b"]


def test_get_parameter_type(grammar: Grammar) -> None:
    assert grammar.get_parameter_type("fn_add", "a") == "number"
    assert grammar.get_parameter_type("fn_greet", "name") == "string"
