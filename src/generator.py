"""Turns a natural-language prompt into a validated FunctionCall.

The heavy lifting (guaranteeing valid JSON / schema-correct types) is
delegated to :mod:`src.constraint_engine`; this module only decides what
to ask the model at each step and assembles the final object.
"""

from src.constraint_engine import ChoiceDecoder, DecodingError, LLM, ValueDecoder
from src.grammar import Grammar
from src.models import Function, FunctionCall
from src.vocabulary import Vocabulary


class GenerationError(Exception):
    """Raised when a prompt cannot be turned into a valid function call."""


class Generator:
    """Generates one :class:`FunctionCall` per user prompt via constrained decoding."""

    def __init__(self, vocabulary: Vocabulary, grammar: Grammar, llm: LLM) -> None:
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
        self._comma_id = vocabulary.get_id(",")
        self._brace_id = vocabulary.get_id("}")
        self._quote_id = vocabulary.get_id('"')

    def generate(self, user_prompt: str) -> FunctionCall:
        """Translate one natural-language prompt into a structured function call.

        Args:
            user_prompt: The natural-language request.

        Returns:
            The resulting :class:`FunctionCall`.

        Raises:
            GenerationError: If no function call could be produced, e.g. an
                empty function set or an unsupported parameter type.
        """
        if not self._grammar.functions:
            raise GenerationError("No functions are declared in functions_definition.json")

        try:
            function_name = self._choose_function_name(user_prompt)
            parameters = self._choose_parameters(user_prompt, function_name)
        except DecodingError as e:
            raise GenerationError(str(e)) from e

        return FunctionCall(prompt=user_prompt, name=function_name, parameters=parameters)

    def _encode(self, text: str) -> list[int]:
        tensor = self._llm.encode(text)
        ids = tensor.flatten().tolist()
        return [int(token_id) for token_id in ids]

    def _value_stop_ids(self) -> frozenset[int]:
        """Ids of whatever can immediately follow a parameter value (`,` or `}`)."""
        return frozenset(i for i in (self._comma_id, self._brace_id) if i is not None)

    # -- function selection --------------------------------------------

    def _choose_function_name(self, user_prompt: str) -> str:
        candidates = self._grammar.get_function_names()
        prompt = self._build_function_selection_prompt(user_prompt, self._grammar.functions)
        prompt_ids = self._encode(prompt)
        stop_ids = frozenset({self._quote_id}) if self._quote_id is not None else frozenset()
        return self._choice_decoder.choose(prompt_ids, candidates, stop_ids=stop_ids)

    @staticmethod
    def _build_function_selection_prompt(user_prompt: str, functions: list[Function]) -> str:
        lines = [
            "You are a function-calling assistant.",
            "Pick the single function that best matches the user request below.",
            "Answer with only the function name, nothing else.",
            "",
            "Available functions:",
        ]
        for function in functions:
            params = ", ".join(f"{n}: {p.type}" for n, p in function.parameters.items())
            lines.append(f"- {function.name}({params}): {function.description}")
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
        function = self._grammar.get_function_by_name(function_name)
        parameters: dict[str, int | float | str | bool] = {}
        for param_name, param in function.parameters.items():
            parameters[param_name] = self._choose_value(
                user_prompt, function, param_name, param.type
            )
        return parameters

    def _choose_value(
        self, user_prompt: str, function: Function, param_name: str, param_type: str
    ) -> int | float | str | bool:
        prompt = self._build_value_prompt(user_prompt, function, param_name, param_type)
        prompt_ids = self._encode(prompt)
        stop_ids = self._value_stop_ids()

        if param_type == "integer":
            return self._value_decoder.generate_number(prompt_ids, integer=True, stop_ids=stop_ids)
        if param_type == "number":
            return self._value_decoder.generate_number(prompt_ids, integer=False, stop_ids=stop_ids)
        if param_type == "string":
            return self._value_decoder.generate_string(prompt_ids)
        if param_type == "boolean":
            choice = self._choice_decoder.choose(prompt_ids, ["true", "false"])
            return choice == "true"

        raise GenerationError(
            f"Unsupported parameter type '{param_type}' for parameter '{param_name}' "
            f"of function '{function.name}'"
        )

    @staticmethod
    def _build_value_prompt(
        user_prompt: str, function: Function, param_name: str, param_type: str
    ) -> str:
        lines = [
            "You extract a single argument value from a user request.",
            "Answer with only the raw value: no quotes, no punctuation, no explanation.",
            "",
            'Example: request "Add 40 and 2", function fn_add_numbers, '
            "parameter a (number) -> 40",
            'Example: request "Say hi to Paul", function fn_greet, '
            "parameter name (string) -> Paul",
            "",
            f'User request: "{user_prompt}"',
            f"Function: {function.name} - {function.description}",
            f"Parameter: {param_name} (type: {param_type})",
            f"{param_name} =",
        ]
        return "\n".join(lines)
