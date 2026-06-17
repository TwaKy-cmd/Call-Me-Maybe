from pydantic import BaseModel

class Parameter(BaseModel):
    type: str


class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class PromptInput(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, int | float | str | bool]