"""Tests for src.__main__: CLI argument parsing, I/O and error handling."""

import json

import pytest

from src.__main__ import (
    InputFileError,
    load_json_list,
    parse_args,
    write_results,
)
from src.models import FunctionCall, PromptInput


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.functions_definition == "data/input/functions_definition.json"
    assert args.input == "data/input/function_calling_tests.json"
    assert args.output == "data/output/function_calling_results.json"


def test_parse_args_overrides() -> None:
    args = parse_args(
        [
            "--functions_definition",
            "custom_functions.json",
            "--input",
            "custom_input.json",
            "--output",
            "custom_output.json",
        ]
    )
    assert args.functions_definition == "custom_functions.json"
    assert args.input == "custom_input.json"
    assert args.output == "custom_output.json"


def test_load_json_list_missing_file_raises(tmp_path) -> None:
    with pytest.raises(InputFileError):
        load_json_list(str(tmp_path / "missing.json"), PromptInput)


def test_load_json_list_invalid_json_raises(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not valid json")
    with pytest.raises(InputFileError):
        load_json_list(str(path), PromptInput)


def test_load_json_list_schema_mismatch_raises(tmp_path) -> None:
    path = tmp_path / "wrong_schema.json"
    path.write_text(json.dumps([{"unexpected_key": "value"}]))
    with pytest.raises(InputFileError):
        load_json_list(str(path), PromptInput)


def test_load_json_list_valid_file(tmp_path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps([{"prompt": "Greet shrek"}, {"prompt": "Reverse 'hi'"}])
    )
    prompts = load_json_list(str(path), PromptInput)
    assert [p.prompt for p in prompts] == ["Greet shrek", "Reverse 'hi'"]


def test_write_results_creates_parent_directories(tmp_path) -> None:
    output_path = tmp_path / "nested" / "output" / "results.json"
    results = [
        FunctionCall(
            prompt="Add 2 and 3",
            name="fn_add_numbers",
            parameters={"a": 2.0, "b": 3.0},
        )
    ]

    write_results(str(output_path), results)

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == [
        {
            "prompt": "Add 2 and 3",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        }
    ]


def test_write_results_handles_empty_list(tmp_path) -> None:
    output_path = tmp_path / "results.json"
    write_results(str(output_path), [])
    assert json.loads(output_path.read_text()) == []
