"""End-to-end tests for src.generator: prompt -> validated FunctionCall."""

import pytest

from src.generator import GenerationError, Generator
from src.grammar import Grammar
from src.models import Function, Parameter
from tests.conftest import OracleLLM


@pytest.fixture
def grammar() -> Grammar:
    functions = [
        Function(
            name="add",
            description="Add two numbers together.",
            parameters={
                "a": Parameter(type="number"),
                "b": Parameter(type="number"),
            },
            returns=Parameter(type="number"),
        ),
        Function(
            name="greet",
            description="Greet a person by name.",
            parameters={"name": Parameter(type="string")},
            returns=Parameter(type="string"),
        ),
    ]
    return Grammar(functions=functions)


def test_generate_picks_function_and_number_parameters(
    vocabulary, grammar: Grammar
) -> None:
    stop_ids = {
        vocabulary.get_id(","),
        vocabulary.get_id("}"),
        vocabulary.get_id('"'),
    }
    llm = OracleLLM(
        vocabulary,
        targets={"Function name:": "add", '"a": ': "4", '"b": ': "2"},
        stop_ids=stop_ids,
    )
    generator = Generator(vocabulary, grammar, llm)

    result = generator.generate("What is the sum of 4 and 2?")

    assert result.prompt == "What is the sum of 4 and 2?"
    assert result.name == "add"
    assert result.parameters == {"a": 4.0, "b": 2.0}


def test_generate_picks_function_and_string_parameter(
    vocabulary, grammar: Grammar
) -> None:
    stop_ids = {
        vocabulary.get_id(","),
        vocabulary.get_id("}"),
        vocabulary.get_id('"'),
    }
    llm = OracleLLM(
        vocabulary,
        targets={"Function name:": "greet", '"name": "': "bob"},
        stop_ids=stop_ids,
    )
    generator = Generator(vocabulary, grammar, llm)

    result = generator.generate("Say hi to bob")

    assert result.name == "greet"
    assert result.parameters == {"name": "bob"}


def test_generate_raises_when_no_functions_are_declared(vocabulary) -> None:
    class UnusedLLM:
        def get_logits_from_input_ids(self, input_ids):
            return []

        def encode(self, text):
            raise AssertionError

    generator = Generator(vocabulary, Grammar(functions=[]), UnusedLLM())
    with pytest.raises(GenerationError):
        generator.generate("anything")
