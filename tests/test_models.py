"""Tests for src.models: pydantic validation of the JSON schemas."""

import pytest
from pydantic import TypeAdapter, ValidationError

from src.models import Function, FunctionCall, PromptInput


def test_function_parses_valid_definition() -> None:
    data = {
        "name": "fn_add_numbers",
        "description": "Add two numbers together.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    }
    function = Function.model_validate(data)
    assert function.name == "fn_add_numbers"
    assert function.parameters["a"].type == "number"


def test_function_missing_field_raises() -> None:
    data = {"name": "fn_add_numbers", "parameters": {}, "returns": {"type": "number"}}
    with pytest.raises(ValidationError):
        Function.model_validate(data)


def test_function_list_from_raw_json() -> None:
    data = [
        {
            "name": "fn_greet",
            "description": "Greet someone.",
            "parameters": {"name": {"type": "string"}},
            "returns": {"type": "string"},
        }
    ]
    functions = TypeAdapter(list[Function]).validate_python(data)
    assert len(functions) == 1
    assert functions[0].name == "fn_greet"


def test_prompt_input_requires_prompt_key() -> None:
    with pytest.raises(ValidationError):
        PromptInput.model_validate({"not_prompt": "hello"})


def test_function_call_accepts_mixed_parameter_types() -> None:
    call = FunctionCall(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={"a": 2.0, "b": 3.0, "label": "sum", "verbose": True},
    )
    dumped = call.model_dump()
    assert dumped["parameters"]["a"] == 2.0
    assert dumped["parameters"]["verbose"] is True
