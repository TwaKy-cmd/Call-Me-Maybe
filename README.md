*This project has been created as part of the 42 curriculum by khebert.*

# call me maybe

## Description

**call me maybe** turns a natural-language request ("What is the sum of 2 and 3?")
into a structured, machine-executable function call
(`{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}`) using a small,
locally-run 0.6B-parameter language model (Qwen3-0.6B).

Small models are notoriously unreliable at free-form JSON generation. Instead of
prompting the model and hoping it produces valid, schema-compliant JSON, this
project implements **constrained decoding from scratch**: at every token the model
is about to generate, the set of next tokens is restricted, using the model's own
logits, to only those that keep the output both syntactically valid and compliant
with the function's schema (`functions_definition.json`). The result is 100% valid,
parseable JSON on every single run, regardless of what the model "wants" to say.

The project has two moving parts:

* `src/constraint_engine.py` — the actual constrained-decoding primitives
  (a token trie for closed-set choices, and grammar-aware masks for open-ended
  values).
* `src/generator.py` — decides *what* to ask the model (which function? which
  value for which parameter?) and assembles the final, schema-correct object.

## Instructions

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. All
commands below assume `llm_sdk/` is present next to `src/` (already the case in
this repository).

```bash
make install   # uv sync — installs numpy, pydantic and llm_sdk
make run       # uv run python -m src, using the default data/input files
make debug     # runs the program under pdb
make lint      # flake8 + mypy (mandatory flags from the subject)
make lint-strict  # flake8 + mypy --strict
make test      # uv run pytest — the (ungraded) unit-test suite
make clean     # removes caches (__pycache__, .mypy_cache, ...)
make fclean    # clean + removes .venv and data/output
make re        # fclean + install, from scratch
```

Running the program directly, with custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

All three flags are optional; they default to the paths shown above. On its very
first run, the LLM SDK downloads the Qwen3-0.6B weights (~1.2 GB) from the Hugging
Face Hub and caches them locally — this can take a few minutes depending on your
connection; subsequent runs load instantly from the local cache.

### A note on the pinned torch build

`pyproject.toml` pins `torch==2.8.0` from the CUDA 12.6 index. The current default
wheels only ship kernels for `sm_75` and above, which excludes Pascal GPUs
(GTX 10xx, `sm_61`) — on such a card the model loads and then dies with
`CUDA error: no kernel image is available for execution on the device`. The cu126
wheels still ship `sm_60` cubins, which run on `sm_61` by CUDA's minor-version
binary compatibility, and cover everything up to `sm_90`. If you are running a
Blackwell card (RTX 50xx, `sm_120`), drop the pin and the `[[tool.uv.index]]`
block to get the default wheels back. `accelerate` is a required dependency
because the SDK passes `device_map="auto"` whenever CUDA is available.

## Resources

* [Hugging Face — Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
* [Hugging Face `transformers` documentation](https://huggingface.co/docs/transformers)
* Willard & Louf, *"Efficient Guided Generation for Large Language Models"* (the
  paper behind the `outlines` library) — the general idea of masking logits with a
  finite-state machine to guarantee grammar-valid output.
* [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259) — used as
  the reference grammar for what counts as a syntactically valid JSON number/string.
* Byte-level BPE tokenization: [the original GPT-2 paper's tokenizer](https://github.com/openai/gpt-2)
  explains why printable ASCII characters map to themselves in the vocab file,
  which this project relies on.

### How AI was used

An AI assistant (Claude) was used throughout this project as a pair-programmer,
under the constraints described in the subject's "AI Instructions" chapter:

* **Architecture discussion**: deciding how to split the problem into a closed-set
  decoder (function name, booleans) driven by a token trie, versus an open-ended
  decoder (numbers, strings) driven by a small per-type grammar — and why literal
  JSON punctuation never needs to go through the model at all (see *Design
  decisions* below).
* **Implementation**: writing the constrained-decoding primitives
  (`src/constraint_engine.py`), the vocabulary/grammar masks (`src/vocabulary.py`),
  the orchestration layer (`src/generator.py`) and the CLI (`src/__main__.py`).
* **Debugging the target environment**: diagnosing that `import torch` was
  pathologically slow in the sandboxed shell used during development (see
  *Challenges faced*), and confirming the real vocabulary file format (byte-level
  BPE) by actually loading the SDK against the cached Qwen3-0.6B weights.
* **Test suite**: designing a fake-LLM test harness (`tests/conftest.py`) that
  drives the constrained decoders deterministically without needing the real model,
  and writing the pytest suite around it.
* **Documentation**: drafting this README.

Every generated line was read, run against the real model, and adjusted based on
real output (see the empty-string/comma bug described in *Challenges faced* below,
found and fixed after actually running the test suite). Nothing here is copy-pasted
blindly — the constrained-decoding design in particular required a genuine
back-and-forth to get the JSON-grammar and stop-condition logic correct.

## Algorithm explanation

Constrained decoding here has two complementary techniques, both built directly on
top of `Small_LLM_Model.get_logits_from_input_ids` — never on the model "just
producing" correct output on its own.

### 1. Closed-set choices — `ChoiceDecoder` (function name, booleans)

A function name (or `true`/`false`) is picked from a small, known set of literal
strings. Each candidate is tokenized with the model's own tokenizer
(`llm.encode`), and all candidates are assembled into a **token-id trie**:

```
root ── "fn_" ── "add" ── "_numbers"   -> fn_add_numbers
              └─ "greet"               -> fn_greet
```

Generation is a walk down this trie:

* At a branching node, the model's logits are masked down to *only* the trie's
  current children, and the highest-scoring one is picked — a genuine decision made
  by the LLM's own probabilities, never by string heuristics.
* If a node has a single child, there is no decision to make, so the token is
  appended directly and **no LLM call happens at all** — most of the time, once the
  first distinguishing character of a function name is chosen, the rest is a free,
  zero-cost walk.
* If a node completes a candidate but also has children (one candidate is a prefix
  of another, e.g. `"hi"` / `"hidden"`), the "keep going" children compete against
  the token that would follow the value in the final JSON (the closing quote) —
  letting the model itself decide whether the shorter or the longer candidate is
  meant.

The walk always ends on a real candidate: it is structurally impossible to return
anything that isn't one of the valid function names.

### 2. Open-ended values — `ValueDecoder` (numbers, strings)

Numbers and strings do not come from a closed set, so they are generated
token-by-token against a tiny per-type grammar, built once from the vocabulary file
at startup (see `src/vocabulary.py`):

* **Numbers** follow a 5-state DFA matching the JSON number grammar
  (`-?[0-9]+(\.[0-9]+)?`): `START → (AFTER_SIGN) → INT_DIGITS → (AFTER_DOT →
  FRAC_DIGITS)`. At every state, only the vocabulary tokens whose *entire* text
  keeps the DFA valid are allowed (a multi-character BPE token like `"42"` is
  accepted or rejected as a whole, not split into two decisions). For the
  `"integer"` type, transitions into the fractional part are simply excluded from
  the allowed set — an integer value is structurally incapable of containing a
  `.`.
* **Strings** allow any vocabulary token whose decoded *bytes* are all printable
  (`>= 0x20`) and contain neither `"` nor `\` — the only bytes that could break
  JSON string syntax. The test is done on decoded bytes, not on the raw
  vocabulary key: in a byte-level BPE vocabulary a newline is stored as the
  printable placeholder `Ċ`, so a naive check on the raw key lets control
  characters straight through (see *Challenges faced*).

For both, the interesting problem is not "what's a valid next character" (that's a
simple mask) but **"when is the value done?"**. A number or a string has no natural
end-of-sequence token. The trick: once the value is already syntactically complete
(at least one digit for a number; any point at all for a string, including empty),
the "keep going" tokens are made to *compete*, using the same logits, against the
token(s) that would immediately follow the value in the final JSON:

* for a number, any token *starting with* a comma or a closing brace;
* for a string, any token *starting with* a quote (comma/brace are deliberately
  **not** used here — see *Challenges faced*, they are legal string content and
  would create an ambiguity).

"Starting with" rather than "equal to" is essential, not a detail: a byte-level
BPE tokenizer merges a closing delimiter with what follows it, so the token that
actually ends a string mid-object is `","`, not `"`. There are 424 tokens
beginning with a quote in Qwen's vocabulary; treating only the bare quote as the
stop signal makes the decoder deaf to what the model is asking for (see
*Challenges faced*).

If the model's own logits favor that "next" token over continuing the value, the
decoder stops there — the model decides its own value's length, while the decoder
guarantees it is never syntactically wrong regardless of that choice. A hard
character/token cap (`max_chars` / `max_tokens`) guarantees termination even in a
pathological case.

### 3. Why the JSON punctuation is never generated by the model

Braces, key names, colons and commas carry **zero decision** for the model — there
is only ever one valid choice at those positions, fully determined by
`functions_definition.json`. `src/generator.py` therefore builds the final
`FunctionCall` object directly in Python (`FunctionCall(prompt=..., name=...,
parameters={...})`) and lets `json.dump` serialize it. JSON validity is guaranteed
*by construction*, not by asking the model to spell out `{`, `"`, `:`... — which
would only burn LLM calls without ever representing a real choice. Constrained
decoding is spent exactly where, and only where, the model actually decides
something: which function, and what value for each argument.

## Design decisions

* **The prompt *is* a partial JSON object.** Rather than describing the value it
  wants in prose, `Generator` writes the JSON it has already committed to and
  asks the model to continue it: `... JSON arguments: {"a": 2.0, "b": `. This one
  decision buys two properties at once — the closing quote/comma the decoder
  competes against becomes the model's own natural next token (a workable stop
  signal instead of an impossible one), and each argument is conditioned on the
  arguments already extracted instead of being decoded in isolation. Both were
  real bugs; see *Challenges faced*.
* **No chat template.** Qwen3-0.6B ships as an instruction-tuned model with a
  ChatML template, but `Small_LLM_Model` only exposes plain `encode(text: str)`
  (no `apply_chat_template`, and reaching into the tokenizer directly would mean
  using a private attribute, which is forbidden). Prompts are therefore plain,
  few-shot, completion-style text. This is slightly less accurate than a proper
  chat template would be, but it is portable to any causal LM the SDK might wrap,
  matching the subject's requirement to support models other than Qwen3-0.6B.
* **Masks precomputed once, not per token.** `Vocabulary` builds every grammar mask
  (string-safe tokens, the 5-state number DFA) a single time at startup by scanning
  the ~150k-token vocabulary once. Per-token generation steps then only do cheap
  set operations, which is what keeps the whole pipeline well inside the 5-minute
  budget even though a real forward pass is not free.
* **`type: ignore[valid-type]` at one spot.** Generic JSON loading
  (`load_json_list(path, model_type)`) uses `TypeAdapter(list[model_type])`, a
  pattern mypy cannot verify statically even though it is correct and exercised by
  tests. This is the single explicit suppression in the codebase.
* **Pydantic everywhere data crosses a boundary.** `Function`, `Parameter`,
  `PromptInput` and `FunctionCall` (`src/models.py`) validate both input files and
  the output; `Grammar` is itself a `BaseModel`, so its `list[Function]` is
  re-validated on construction; and `Vocabulary` validates the raw vocabulary file
  through a pydantic `TypeAdapter(dict[str, int])` rather than hand-rolled
  `isinstance` checks. Together this turns any malformed input into one clear
  `InputFileError` or `ValueError` instead of a raw traceback.
  The classes that are *not* pydantic models are the ones that carry no external
  data to validate: exceptions (`InputFileError`, `DecodingError`,
  `GenerationError`), the structural `LLM` `Protocol`, the `NumberState` enum, the
  internal `_TrieNode`, and the three behaviour-only classes (`ChoiceDecoder`,
  `ValueDecoder`, `Generator`) whose only state is a model handle and precomputed
  masks. Validating those would cost per-token overhead on the hot path without
  checking anything a type annotation does not already guarantee.
* **A `Protocol`, not a concrete `Small_LLM_Model` import, inside the engine.**
  `src/constraint_engine.py` and `src/generator.py` depend on a small structural
  `LLM` protocol (`get_logits_from_input_ids`, `encode`), not on `llm_sdk` itself.
  Only `src/__main__.py` imports the real SDK. This keeps the decoding logic fully
  unit-testable with a fake model (see *Testing strategy*) and is why the test
  suite runs in well under a second instead of needing the real 0.6B model.

## Performance analysis

* **JSON validity: 100%**, always — guaranteed structurally (see *Algorithm
  explanation*, point 3), not empirically measured, since the output is assembled
  from typed Python values rather than parsed back out of generated text.
* **Accuracy** (right function, right argument values) depends on the small
  model's language understanding, which constrained decoding does not — and
  cannot — improve; it only guarantees the *shape* of the answer. The generation
  logic itself was validated two ways: (1) the full `ChoiceDecoder` /
  `ValueDecoder` code paths — masking, the trie walk, the stop-vs-continue
  competition — are exercised deterministically by the unit-test suite against a
  fake model (see *Testing strategy*), and (2) the real `Small_LLM_Model` was
  loaded against the actual cached Qwen3-0.6B weights to confirm the SDK
  integration itself: `encode`, `get_path_to_vocab_file` and
  `get_logits_from_input_ids` all behave exactly as assumed by
  `src/vocabulary.py` and `src/constraint_engine.py` (byte-level BPE tokens,
  printable ASCII mapping to itself, a `list[float]` logits vector per call).
  Measured on the 11 bundled prompts against the real Qwen3-0.6B:
  **function selection 11/11 (100%)**, argument values **18/19 (95%)**, for
  **10/11 prompts entirely correct**. The single miss is `" *"` where `"*"` was
  meant — the model picked the vocabulary token that carries a leading space.
  That leading space is deliberately *not* stripped: a space is a perfectly
  legitimate argument value (replacing spaces with dashes is a realistic call),
  so trimming it would trade a cosmetic win for a correctness bug.
  Every regex, number, name and quoted substring is extracted correctly.
  One honest caveat on how that number was reached: the third few-shot example in
  the argument prompt (a space/dash replacement) was added *after* observing that
  the model echoed `"asterisk"` instead of `"*"`. The example itself is generic —
  it teaches "when the request names a character, write the character", and none
  of the bundled prompts appear in it — but it was chosen in response to a
  measured failure, so treat 95% as the accuracy on *this* prompt set, not as a
  guaranteed floor on the reviewer's.
  `data/output/function_calling_results.json` is intentionally not committed
  (see *Submission* rules), so run `make run` to reproduce these figures.
* **Speed**: **31s** for the full 11-prompt run on a laptop GTX 1050, of which a
  few seconds are the one-time model load — comfortably inside the 5-minute
  budget. Once loaded, each decision needs anywhere from a single LLM call (an
  unambiguous function-name prefix, via the trie's single-child shortcut) to a
  few dozen (a long string or number, one call per generated token). Note that
  `get_logits_from_input_ids` re-runs the whole prefix on every call (the SDK
  exposes no KV cache), so cost grows with prompt length. Two things brought the
  run down from an initial 183s: applying the logit masks with a single
  vectorized NumPy scatter instead of a Python loop over the ~150k allowed ids,
  and fixing the stop set (see *Challenges faced*) — a value that stops when the
  model wants it to costs a handful of forward passes instead of the full
  `max_tokens` cap. Without a usable GPU the same run takes well over 15 minutes.

## Challenges faced

* **Vocabulary keys are not the text they stand for.** The first real run
  produced `{"name": "ĠshrekĊAnswer:Ġname..."}`. In a byte-level BPE vocabulary
  every raw byte is re-encoded into a printable placeholder — a space is stored
  as `Ġ`, a newline as `Ċ` — so reading `vocab.json` keys as literal text leaks
  those placeholders straight into the output. Worse, it silently broke the
  string mask: the check excluded the character `"\n"`, which simply never
  appears in such a vocabulary, so *newline tokens were allowed inside JSON
  strings*. `Vocabulary` now rebuilds the inverse byte table once at import
  (`_byte_decoder`) and every mask works on decoded `bytes`. Tokens are
  accumulated as `bytes` and decoded only once the value is complete, because a
  single token can hold a fragment of a multi-byte UTF-8 character (a `€` is
  three bytes and may be split).
* **A prompt that contradicted its own stop signal.** Strings stop when the model
  prefers the closing quote — but the value prompt used to end with `name =` and
  instructed "no quotes, no punctuation". The quote was therefore the one token
  the model would never pick, and every string ran to the `max_tokens` cap. The
  fix was to stop *describing* the target and start *writing* it: the prompt now
  ends with a partial JSON object whose opening quote is already written
  (`{"name": "`), making the closing quote the natural continuation.
* **The stop signal was one token wide, and the model never used it.** Even with
  the partial-JSON prompt above, three prompts still produced runaway values
  (`cat.*cat.*cat.*...`) up to the `max_tokens` cap. The tempting conclusion was
  "a 0.6B model cannot synthesize a regex" — dumping the raw logits at each step
  proved otherwise. From the very first token the model's **top-1 choice overall**
  was `","` (logit 24.6), i.e. *close the string and move to the next argument*.
  But `","` contains a quote, so it was excluded from `string_content_ids`; and it
  is not the bare `"` token, so it was not in the stop set either. Masked to
  `-inf` from both sides, it was discarded, and the decoder fell back to the best
  *content* token (`.*`, logit 19.7) — over and over. The bare quote sat at rank
  121. The lesson: with byte-level BPE, a delimiter is almost never a token of its
  own, so a stop set must be defined by *prefix*, not by equality. Diagnosing this
  meant printing the top-k logits rather than trusting the plausible explanation.
* **Arguments extracted in isolation repeated each other.** Each parameter used
  to get its own independent prompt, so "What is the sum of 2 and 3?" yielded
  `{"a": 2, "b": 2}` — asked for `b` in isolation, the model has no reason not to
  answer with the first number it sees. Feeding back the already-decoded
  parameters as a partial JSON prefix (`{"a": 2.0, "b": ` ) fixes it, and falls
  out of the same design as the point above.
* **`import torch` hanging in the sandboxed development shell.** Early testing
  showed `Small_LLM_Model()` never returning — 0% CPU usage for minutes on end.
  Bisecting the SDK step by step (plain `import torch`, then tokenizer loading,
  then model loading, then `.to(device)`) showed the *import itself* was the
  bottleneck (490s+ in the restricted shell, vs. a normal few seconds), a property
  of that specific sandbox, not of the code. Every real-model test in this project
  was therefore run with that restriction lifted; nothing in `src/` depends on it.
* **Comma/brace as a string "stop" signal was ambiguous.** The first version of
  `ValueDecoder.generate_string` reused the same "compete against the next literal
  token" trick used for numbers, offering the comma and closing-brace tokens as a
  stop signal. But a comma is *also* perfectly legal string content (`"Hello,
  world"`), so the decoder could not tell "the model wants to stop" from "the model
  wants a comma in the string" — caught by the unit tests
  (`tests/test_constraint_engine.py`), which failed with a string trailing off into
  garbage instead of stopping. Fixed by observing that a JSON string is
  unambiguously delimited by its closing quote alone; only the quote token (which
  can never be legal, unescaped string content) is now used as the stop signal for
  strings, while numbers keep using comma/brace (which, unlike for strings, can
  never legally appear *inside* a bare JSON number).
* **Multi-character BPE tokens inside a strict grammar.** A number-safe token like
  `"-42"` or `"3."` must be validated as a whole against the DFA (rejecting it
  outright if any character inside it would violate the number grammar), not
  character-by-character after the fact — otherwise a single token could jump the
  generation into an invalid state that is only noticed after it has already been
  appended. `Vocabulary._build_number_masks` walks every token's full text through
  the DFA at load time and only keeps tokens that stay valid all the way through.

## Testing strategy

The bundled `tests/` suite (via `make test` / `uv run pytest`, ungraded per the
subject) does **not** load the real 0.6B model — that would make the suite slow
and non-deterministic. Instead, `tests/conftest.py` provides:

* a tiny, fully deterministic character-level vocabulary (one token per printable
  ASCII character), covering every character any prompt built by `Generator` can
  contain;
* `OracleLLM`, a fake model that decodes `input_ids` back to text, recognizes which
  prompt it is being asked to continue, and boosts the logits of whatever
  character would extend a hidden target string — until the target is complete, at
  which point it boosts the configured "stop" tokens instead. This drives the real
  `ChoiceDecoder`/`ValueDecoder` code paths (masking, the trie walk, the
  stop-vs-continue competition) exactly as the real SDK would, deterministically.

This caught a real bug (the comma/string ambiguity above) before it ever reached
the real model. On top of the constrained-decoding engine, the suite covers:
`Vocabulary`'s grammar masks and error handling, `Grammar`'s lookups, `models.py`'s
pydantic validation, `Generator`'s end-to-end assembly, and `src/__main__.py`'s
file loading / CLI parsing / output writing.

Separately, the pipeline was validated against the **real** Qwen3-0.6B model by
actually running `make run` against the bundled `data/input/` fixtures and
inspecting `data/output/function_calling_results.json` for JSON validity, correct
types, and correct function/argument selection.

## Example usage

```bash
$ make run
Loading the language model (this can take a while on first run)...
Wrote 11/11 function call(s) to data/output/function_calling_results.json
```

```bash
$ uv run python -m src --input data/input/function_calling_tests.json
```

Given `functions_definition.json` declaring `fn_add_numbers(a: number, b: number)`
and the prompt `"What is the sum of 2 and 3?"`, `data/output/function_calling_results.json`
contains:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

Missing or malformed input files fail with a clear message instead of a traceback:

```bash
$ uv run python -m src --input data/input/does_not_exist.json
Error: file not found: data/input/does_not_exist.json
```
