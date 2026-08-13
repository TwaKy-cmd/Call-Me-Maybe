"""CLI entry point: turns natural-language prompts into function calls.

Usage:
    uv run python -m src [--functions_definition PATH] [--input PATH] [--output PATH]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from src.generator import GenerationError, Generator
from src.grammar import Grammar
from src.models import Function, FunctionCall, PromptInput
from src.vocabulary import Vocabulary

DEFAULT_FUNCTIONS_DEFINITION = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"

ModelT = TypeVar("ModelT", bound=BaseModel)


class InputFileError(Exception):
    """Raised when an input file is missing, malformed or invalid."""


def load_json_list(path: str, model_type: type[ModelT]) -> list[ModelT]:
    """Load and validate a JSON array of `model_type` objects.

    Args:
        path: Path to the JSON file.
        model_type: The pydantic model each array element must match.

    Returns:
        The list of validated model instances.

    Raises:
        InputFileError: If the file is missing, not valid JSON, or does
            not match the expected schema.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise InputFileError(f"file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise InputFileError(f"invalid JSON in {path}: {e}") from e

    try:
        return TypeAdapter(list[model_type]).validate_python(data)  # type: ignore[valid-type]
    except ValidationError as e:
        raise InputFileError(f"{path} does not match the expected schema: {e}") from e


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, applying the documented defaults."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into structured function calls.",
    )
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS_DEFINITION)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def write_results(path: str, results: list[FunctionCall]) -> None:
    """Write the generated function calls as a JSON array.

    Args:
        path: Destination file path; parent directories are created if
            needed.
        results: The function calls to serialize.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump() for result in results]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline and return a process exit code."""
    args = parse_args(argv)

    try:
        functions = load_json_list(args.functions_definition, Function)
        prompts = load_json_list(args.input, PromptInput)
    except InputFileError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as e:
        print(f"Error: could not import the LLM SDK: {e}", file=sys.stderr)
        return 1

    print("Loading the language model (this can take a while on first run)...", file=sys.stderr)
    try:
        llm = Small_LLM_Model()
    except Exception as e:  # noqa: BLE001 - SDK/torch/HF can fail in many ways
        print(f"Error: could not load the language model: {e}", file=sys.stderr)
        return 1

    try:
        vocabulary = Vocabulary(llm.get_path_to_vocab_file())
    except (OSError, ValueError) as e:
        print(f"Error: could not load the model vocabulary: {e}", file=sys.stderr)
        return 1

    grammar = Grammar(functions)
    generator = Generator(vocabulary, grammar, llm)

    results: list[FunctionCall] = []
    for prompt_input in prompts:
        try:
            results.append(generator.generate(prompt_input.prompt))
        except (GenerationError, ValueError) as e:
            print(f"Warning: skipped prompt {prompt_input.prompt!r}: {e}", file=sys.stderr)

    write_results(args.output, results)
    print(f"Wrote {len(results)}/{len(prompts)} function call(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
