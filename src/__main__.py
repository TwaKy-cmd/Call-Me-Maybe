import sys
import json
from src.models import Function, PromptInput
from pydantic import TypeAdapter, BaseModel, ValidationError

def load_json_file(path: str, model_type: type[BaseModel]) -> list:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error : {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error : {e}")
        sys.exit(1)

    try:
        obj = TypeAdapter(list[model_type]).validate_python(data)
    except ValidationError as e:
        print(f"Error : {e}")
        sys.exit(1)

    return obj


def main():
    functions = load_json_file("data/input/functions_definition.json", Function)
    for func in functions:
        print(f"Fonction chargée : {func.name}")

    prompts = load_json_file("data/input/function_calling_tests.json", PromptInput)
    for p in prompts:
        print(f"Prompt chargé : {p.prompt}")



if __name__ == "__main__":
    main()