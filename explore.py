import os
os.environ["HF_HUB_VERBOSITY"] = "info"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"

print("Avant import...")
from llm_sdk import Small_LLM_Model
print("Import OK")