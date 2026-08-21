"""Turns a natural-language prompt into a validated FunctionCall.

The heavy lifting (guaranteeing valid JSON / schema-correct types) is
delegated to :mod:`src.constraint_engine`; this module only decides what
to ask the model at each step and assembles the final object.
"""

import json

from src.constraint_engine import (
    ChoiceDecoder,
    DecodingError,
    LLM,
    ValueDecoder,
)
from src.grammar import Grammar
from src.models import Function, FunctionCall
from src.vocabulary import Vocabulary


class GenerationError(Exception):
    """Raised when a prompt cannot be turned into a valid function call."""


class Generator:
    """Generates one :class:`FunctionCall` per prompt via constrained
    decoding."""

    def __init__(
        self, vocabulary: Vocabulary, grammar: Grammar, llm: LLM
    ) -> None:
        """Wire together the vocabulary, the function schema and the LLM.

        Args:
            vocabulary: Precomputed token vocabulary and grammar masks.
            grammar: The available functions to choose from.
            llm: The language model used for every decoding decision.
        """
        self._vocab = vocabulary
        self._grammar = grammar
        self._llm = llm
        self._choice_decoder = ChoiceDecoder(llm)
        self._value_decoder = ValueDecoder(llm, vocabulary)
        self._quote_id = vocabulary.get_id('"')
        # Anything starting with a comma or a closing brace means "this
        # value is finished" -- see Vocabulary.ids_starting_with.
        self._number_stop_ids = vocabulary.ids_starting_with(
            ","
        ) | vocabulary.ids_starting_with("}")

    def generate(self, user_prompt: str) -> FunctionCall:
        """Translate one natural-language prompt into a structured
        function call.

        Args:
            user_prompt: The natural-language request.

        Returns:
            The resulting :class:`FunctionCall`.

        Raises:
            GenerationError: If no function call could be produced, e.g. an
                empty function set or an unsupported parameter type.
        """
        if not self._grammar.functions:
            raise GenerationError(
                "No functions are declared in functions_definition.json"
            )

        try:
            function_name = self._choose_function_name(user_prompt)
            parameters = self._choose_parameters(user_prompt, function_name)
        except DecodingError as e:
            raise GenerationError(str(e)) from e

        return FunctionCall(
            prompt=user_prompt, name=function_name, parameters=parameters
        )

    def _encode(self, text: str) -> list[int]:
        tensor = self._llm.encode(text)
        ids = tensor.flatten().tolist()
        return [int(token_id) for token_id in ids]

    def _value_stop_ids(self) -> frozenset[int]:
        """Ids that can immediately follow a parameter value (`,` or `}`)."""
        return self._number_stop_ids

    # -- function selection --------------------------------------------

    def _choose_function_name(self, user_prompt: str) -> str:
        candidates = self._grammar.get_function_names()
        prompt = self._build_function_selection_prompt(
            user_prompt, self._grammar.functions
        )
        prompt_ids = self._encode(prompt)
        stop_ids = (
            frozenset({self._quote_id})
            if self._quote_id is not None
            else frozenset()
        )
        return self._choice_decoder.choose(
            prompt_ids, candidates, stop_ids=stop_ids
        )

    @staticmethod
    def _build_function_selection_prompt(
        user_prompt: str, functions: list[Function]
    ) -> str:
        lines = [
            "You are a function-calling assistant.",
            "Pick the single function that best matches the user "
            "request below.",
            "Answer with only the function name, nothing else.",
            "",
            "Available functions:",
        ]
        for function in functions:
            params = ", ".join(
                f"{n}: {p.type}" for n, p in function.parameters.items()
            )
            lines.append(
                f"- {function.name}({params}): {function.description}"
            )
        lines += [
            "",
            f'User request: "{user_prompt}"',
            "",
            "Function name:",
        ]
        return "\n".join(lines)

    # -- parameter values -------------------------------------------------

    def _choose_parameters(
        self, user_prompt: str, function_name: str
    ) -> dict[str, int | float | str | bool]:
        """Fill the argument object one parameter at a time.

        The parameters already decoded are re-injected into the prompt as
        a partial JSON object, so each new value is conditioned on them.
        Without that, every parameter would be extracted from the very
        same prompt and the model would happily repeat the first value it
        found (``{"a": 2, "b": 2}`` for "the sum of 2 and 3").
        """
        function = self._grammar.get_function_by_name(function_name)
        header = self._build_arguments_header(user_prompt, function)
        parameters: dict[str, int | float | str | bool] = {}
        filled: list[str] = []

        for param_name, param in function.parameters.items():
            prefix = header + self._json_prefix(filled, param_name)
            value = self._choose_value(
                prefix, function, param_name, param.type
            )
            parameters[param_name] = value
            filled.append(f"{json.dumps(param_name)}: {json.dumps(value)}")

        return parameters

    @staticmethod
    def _json_prefix(filled: list[str], param_name: str) -> str:
        """Render the partial JSON object up to the value to decode."""
        return (
            "{"
            + "".join(f"{fragment}, " for fragment in filled)
            + f'"{param_name}": '
        )

    def _choose_value(
        self,
        prefix: str,
        function: Function,
        param_name: str,
        param_type: str,
    ) -> int | float | str | bool:
        """Decode one value, continuing the partial JSON in `prefix`."""
        stop_ids = self._value_stop_ids()

        if param_type == "integer":
            prompt_ids = self._encode(prefix)
            return self._value_decoder.generate_number(
                prompt_ids, integer=True, stop_ids=stop_ids
            )
        if param_type == "number":
            prompt_ids = self._encode(prefix)
            return self._value_decoder.generate_number(
                prompt_ids, integer=False, stop_ids=stop_ids
            )
        if param_type == "string":
            # The opening quote is written for the model, so that closing
            # it is the natural continuation and can serve as stop signal.
            prompt_ids = self._encode(prefix + '"')
            return self._value_decoder.generate_string(prompt_ids)
        if param_type == "boolean":
            prompt_ids = self._encode(prefix)
            choice = self._choice_decoder.choose(prompt_ids, ["true", "false"])
            return choice == "true"

        raise GenerationError(
            f"Unsupported parameter type '{param_type}' for parameter "
            f"'{param_name}' of function '{function.name}'"
        )

    @staticmethod
    def _build_arguments_header(
        user_prompt: str, function: Function
    ) -> str:
        """Everything preceding the partial JSON object in a value prompt."""
        signature = ", ".join(
            f"{n} ({p.type})" for n, p in function.parameters.items()
        )
        lines = [
            "You fill in the arguments of a function call as a JSON "
            "object.",
            "Copy the values from the user request exactly, then close "
            "the JSON.",
            "",
            'Example: request "Add 40 and 2", function fn_add_numbers(a, b)',
            'JSON arguments: {"a": 40, "b": 2}',
            'Example: request "Say hi to Paul", function fn_greet(name)',
            'JSON arguments: {"name": "Paul"}',
            'Example: request "Replace every space with a dash in \'a b c\'", '
            "function fn_replace(source, old, new)",
            'JSON arguments: {"source": "a b c", "old": " ", "new": "-"}',
            "",
            f'User request: "{user_prompt}"',
            f"Function: {function.name} - {function.description}",
            f"Arguments: {signature}",
            "",
            "JSON arguments: ",
        ]
        return "\n".join(lines)
