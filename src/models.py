"""Pydantic data models shared across the function-calling pipeline."""

from pydantic import BaseModel

#: JSON-schema-like parameter types this project knows how to generate
#: through constrained decoding.
SUPPORTED_PARAMETER_TYPES = ("number", "integer", "string", "boolean")


class Parameter(BaseModel):
    """A single typed field, used both for function parameters and returns.

    Attributes:
        type: The JSON-schema-like type name (e.g. "number", "string").
    """

    type: str


class Function(BaseModel):
    """Describes one callable function declared in
    functions_definition.json.

    Attributes:
        name: The function's identifier (e.g. "fn_add_numbers").
        description: Natural-language description used to help the LLM
            pick the right function for a given prompt.
        parameters: Mapping of parameter name to its :class:`Parameter`.
        returns: The type returned by the function (informational only,
            not part of the generated output).
    """

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class PromptInput(BaseModel):
    """A single natural-language request to translate into a function call.

    Attributes:
        prompt: The raw user request.
    """

    prompt: str


class FunctionCall(BaseModel):
    """The structured result written to the output file for one prompt.

    Attributes:
        prompt: The original natural-language request, copied verbatim.
        name: The name of the selected function.
        parameters: The extracted arguments, typed according to the
            function's schema.
    """

    prompt: str
    name: str
    parameters: dict[str, int | float | str | bool]
