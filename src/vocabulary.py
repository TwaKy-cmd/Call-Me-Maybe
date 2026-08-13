"""Token vocabulary loading and precomputed JSON-grammar masks.

The vocabulary file maps each token's literal text (as produced by the
model's byte-level BPE tokenizer) to its integer id. Printable ASCII
characters -- digits, ``.``, ``-``, quotes, JSON punctuation -- map to
themselves under that encoding, so JSON-grammar masks can be built by
inspecting token text directly, once, at load time. Every mask below is
therefore computed a single time in ``__init__`` so that per-token
generation steps only need cheap set lookups, keeping the pipeline fast
even with a ~150k-token vocabulary.
"""

import json
from enum import Enum, auto


class NumberState(Enum):
    """States of the small DFA describing a syntactically valid JSON number."""

    START = auto()
    AFTER_SIGN = auto()
    INT_DIGITS = auto()
    AFTER_DOT = auto()
    FRAC_DIGITS = auto()


_ACCEPTING_NUMBER_STATES = frozenset({NumberState.INT_DIGITS, NumberState.FRAC_DIGITS})


def _number_transition(state: NumberState, char: str) -> NumberState | None:
    """Single-character transition of the JSON-number DFA, or None if invalid."""
    if state is NumberState.START:
        if char == "-":
            return NumberState.AFTER_SIGN
        return NumberState.INT_DIGITS if char.isdigit() else None
    if state is NumberState.AFTER_SIGN:
        return NumberState.INT_DIGITS if char.isdigit() else None
    if state is NumberState.INT_DIGITS:
        if char.isdigit():
            return NumberState.INT_DIGITS
        return NumberState.AFTER_DOT if char == "." else None
    if state is NumberState.AFTER_DOT:
        return NumberState.FRAC_DIGITS if char.isdigit() else None
    if state is NumberState.FRAC_DIGITS:
        return NumberState.FRAC_DIGITS if char.isdigit() else None
    return None


NumberMask = tuple[frozenset[int], dict[int, NumberState]]


class Vocabulary:
    """Maps token text <-> token id and precomputes constrained-decoding masks."""

    _FORBIDDEN_STRING_CHARS = frozenset('"\\\n\r\t')

    def __init__(self, vocab_path: str) -> None:
        """Load the vocabulary file and build every grammar mask once.

        Args:
            vocab_path: Path to the vocab file, as returned by
                ``Small_LLM_Model.get_path_to_vocab_file()``.

        Raises:
            OSError: If the file cannot be opened.
            ValueError: If the file is not a valid {token: id} JSON object.
        """
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except OSError as e:
            raise OSError(f"Could not read vocabulary file '{vocab_path}': {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Vocabulary file '{vocab_path}' is not valid JSON: {e}") from e

        if not isinstance(data, dict) or not data:
            raise ValueError(f"Vocabulary file '{vocab_path}' must contain a non-empty object")
        if not all(isinstance(k, str) and isinstance(v, int) for k, v in data.items()):
            raise ValueError(
                f"Vocabulary file '{vocab_path}' must map token strings to integer ids"
            )

        self.text_to_id: dict[str, int] = data
        self.id_to_text: dict[int, str] = {v: k for k, v in data.items()}

        self.string_content_ids: frozenset[int] = self._build_string_content_ids()
        self._number_masks: dict[NumberState, NumberMask] = self._build_number_masks()

    def get_text(self, token_id: int) -> str:
        """Return the literal text of a token id."""
        return self.id_to_text[token_id]

    def get_id(self, text: str) -> int | None:
        """Return the token id for an exact literal token text, if any."""
        return self.text_to_id.get(text)

    def _build_string_content_ids(self) -> frozenset[int]:
        return frozenset(
            token_id
            for text, token_id in self.text_to_id.items()
            if text and self._FORBIDDEN_STRING_CHARS.isdisjoint(text)
        )

    def _build_number_masks(self) -> dict[NumberState, NumberMask]:
        masks: dict[NumberState, NumberMask] = {}
        for state in NumberState:
            allowed: set[int] = set()
            end_states: dict[int, NumberState] = {}
            for text, token_id in self.text_to_id.items():
                if not text:
                    continue
                current = state
                for char in text:
                    next_state = _number_transition(current, char)
                    if next_state is None:
                        break
                    current = next_state
                else:
                    allowed.add(token_id)
                    end_states[token_id] = current
            masks[state] = (frozenset(allowed), end_states)
        return masks

    def number_mask(self, state: NumberState) -> NumberMask:
        """Return (allowed token ids, resulting end state) to extend a number in `state`."""
        return self._number_masks[state]

    @staticmethod
    def is_accepting_number_state(state: NumberState) -> bool:
        """Whether `state` already forms a syntactically complete JSON number."""
        return state in _ACCEPTING_NUMBER_STATES
