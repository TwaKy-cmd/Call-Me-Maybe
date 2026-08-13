"""Read-only view over the available functions (functions_definition.json)."""

from pydantic import BaseModel

from src.models import Function


class Grammar(BaseModel):
    """Exposes lookups over a set of callable :class:`Function` definitions.

    Attributes:
        functions: The parsed content of functions_definition.json. Pydantic
            re-validates every element here, so a `Grammar` can never be
            built around a malformed function definition.
    """

    functions: list[Function]

    def get_function_names(self) -> list[str]:
        """Return every declared function name, in definition order."""
        return [f.name for f in self.functions]

    def get_function_by_name(self, name: str) -> Function:
        """Look up a function by its exact name.

        Args:
            name: The function name to search for.

        Returns:
            The matching :class:`Function`.

        Raises:
            ValueError: If no function with this name is declared.
        """
        for f in self.functions:
            if f.name == name:
                return f
        raise ValueError(f"Function '{name}' not found")

    def get_parameter_names(self, function_name: str) -> list[str]:
        """Return the ordered parameter names of a function."""
        func = self.get_function_by_name(function_name)
        return list(func.parameters.keys())

    def get_parameter_type(self, function_name: str, param_name: str) -> str:
        """Return the declared type of one parameter of a function."""
        func = self.get_function_by_name(function_name)
        return func.parameters[param_name].type
