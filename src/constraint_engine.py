"""Low-level constrained-decoding primitives.

Two complementary techniques are implemented here:

* :class:`ChoiceDecoder` picks the single best candidate out of a small
  closed set of literal strings (e.g. a function name, or "true"/"false")
  by walking, token by token, a trie built from the *real* tokenization
  of each candidate. At every branching point, the next token is chosen
  by masking the model's logits down to the trie's current children and
  taking the highest-scoring one -- so the choice is genuinely made by
  the LLM's own logits, never by string heuristics.

* :class:`ValueDecoder` generates an open-ended value (a number or a
  string) for a function parameter. At every step it restricts the next
  token to whatever keeps the value both syntactically valid (a
  well-formed JSON number, or JSON-string-safe text) and lets the model
  itself decide when the value is complete, by comparing the best
  "keep going" token against a "stop" signal built from the token(s)
  that would immediately follow the value in the final JSON (a closing
  quote, a comma, a closing brace...).

Every literal, unambiguous piece of the output JSON (braces, key names,
colons, commas...) carries no decision for the model to make, so it is
never sent through the LLM -- it is written directly by
:class:`~src.generator.Generator`. Constrained decoding is spent exactly
where, and only where, the model actually chooses something: which
function to call, and what each argument's value is.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

import numpy as np

from src.vocabulary import NumberState, Vocabulary


class LLM(Protocol):
    """The subset of Small_LLM_Model this module relies on."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        ...

    def encode(self, text: str) -> Any:
        ...


class DecodingError(Exception):
    """Raised when constrained decoding cannot produce any valid token."""


def _as_id_array(ids: Iterable[int] | np.ndarray) -> np.ndarray:
    """Normalize a set of candidate token ids into an int64 array."""
    if isinstance(ids, np.ndarray):
        return ids
    return np.fromiter(ids, dtype=np.int64)


def _mask_logits(
    logits: np.ndarray, allowed_ids: Iterable[int] | np.ndarray
) -> np.ndarray:
    """Return a copy of `logits` with ids outside `allowed_ids` set to -inf.

    The candidate set can hold most of a ~150k-token vocabulary (any token
    is legal inside a JSON string), so the mask is applied with a single
    vectorized scatter rather than a Python-level loop.
    """
    masked = np.full_like(logits, -np.inf)
    ids = _as_id_array(allowed_ids)
    ids = ids[(ids >= 0) & (ids < masked.shape[0])]
    masked[ids] = logits[ids]
    return masked


def _argmax_among(logits: np.ndarray, ids: Iterable[int] | np.ndarray) -> int:
    """Return the id (from `ids`) with the highest logit.

    Raises:
        DecodingError: If none of `ids` fits within `logits`.
    """
    masked = _mask_logits(logits, ids)
    best_id = int(np.argmax(masked))
    if not math.isfinite(masked[best_id]):
        raise DecodingError(
            "No candidate token is representable by this model's "
            "vocabulary"
        )
    return best_id


@dataclass
class _TrieNode:
    """One node of a token-id trie built from tokenized candidate strings."""

    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    candidate: str | None = None


def _encode_ids(llm: LLM, text: str) -> list[int]:
    """Tokenize `text` through the LLM's own tokenizer, as a plain id list."""
    tensor = llm.encode(text)
    ids = tensor.flatten().tolist()
    return [int(token_id) for token_id in ids]


class ChoiceDecoder:
    """Picks one literal string out of a closed set, via real LLM logits."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def choose(
        self,
        prompt_ids: list[int],
        candidates: list[str],
        stop_ids: frozenset[int] = frozenset(),
    ) -> str:
        """Greedily walk the token trie of `candidates` to pick the best one.

        Args:
            prompt_ids: Token ids of the prompt the choice is conditioned on.
            candidates: The closed set of allowed literal strings. Must be
                non-empty.
            stop_ids: Token ids that, when they outscore every trie child at
                a point where a candidate is already complete, make the walk
                stop there instead of continuing towards a longer candidate
                sharing the same prefix.

        Returns:
            The chosen candidate string.
        """
        if not candidates:
            raise ValueError("ChoiceDecoder requires at least one candidate")
        if len(candidates) == 1:
            return candidates[0]

        root = _TrieNode()
        for candidate in candidates:
            node = root
            for token_id in _encode_ids(self._llm, candidate):
                node = node.children.setdefault(token_id, _TrieNode())
            node.candidate = candidate

        node = root
        running_ids = list(prompt_ids)
        while True:
            if node.candidate is not None and not node.children:
                return node.candidate

            if node.candidate is None and len(node.children) == 1:
                # Only one continuation exists: no decision to make,
                # no LLM call needed.
                (only_id, only_child),  = node.children.items()
                running_ids.append(only_id)
                node = only_child
                continue

            logits = np.asarray(
                self._llm.get_logits_from_input_ids(running_ids),
                dtype=np.float64,
            )
            competitors = set(node.children) | (
                stop_ids if node.candidate is not None else set()
            )
            best_id = _argmax_among(logits, competitors)

            if node.candidate is not None and best_id not in node.children:
                return node.candidate

            running_ids.append(best_id)
            node = node.children[best_id]


class ValueDecoder:
    """Generates an open-ended number or string value for a parameter."""

    def __init__(self, llm: LLM, vocabulary: Vocabulary) -> None:
        self._llm = llm
        self._vocab = vocabulary

    def generate_number(
        self,
        prompt_ids: list[int],
        *,
        integer: bool,
        stop_ids: frozenset[int],
        max_chars: int = 32,
    ) -> int | float:
        """Generate a JSON number token by token, respecting its grammar.

        Args:
            prompt_ids: Token ids of the prompt the value is conditioned on.
            integer: If True, never enter the fractional part of the grammar.
            stop_ids: Token ids of whatever immediately follows the number in
                the final JSON (typically a comma or closing brace). Used as
                the model's own signal that the number is already complete.
            max_chars: Hard safety cap on the number of characters, in case
                the model never prefers to stop.

        Returns:
            The generated number, as an int or a float.
        """
        state = NumberState.START
        running_ids = list(prompt_ids)
        text = ""

        while len(text) < max_chars:
            allowed_ids, end_states = self._vocab.number_mask(state)
            if integer:
                allowed_ids = frozenset(
                    token_id
                    for token_id in allowed_ids
                    if end_states[token_id] not in (
                        NumberState.AFTER_DOT, NumberState.FRAC_DIGITS
                    )
                )

            accepting = Vocabulary.is_accepting_number_state(state)
            competitors = set(allowed_ids) | (stop_ids if accepting else set())
            if not competitors:
                break

            logits = np.asarray(
                self._llm.get_logits_from_input_ids(running_ids),
                dtype=np.float64,
            )
            best_id = _argmax_among(logits, competitors)
            if best_id not in allowed_ids:
                break

            running_ids.append(best_id)
            text += self._vocab.get_text(best_id)
            state = end_states[best_id]

        if not text:
            raise DecodingError(
                "Constrained number generation produced no digits"
            )
        return int(text) if integer else float(text)

    def generate_string(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 48,
    ) -> str:
        """Generate the *content* of a JSON string, token by token.

        Unlike numbers, a JSON string is unambiguously delimited by its
        closing quote -- comma or brace characters are ordinary, legal
        string content (e.g. "Hello, world"), so they cannot double as a
        stop signal the way they do for numbers. The closing quote is
        therefore the only token competed against "keep going", which is
        why the caller must condition this on a prompt whose last written
        character is the string's *opening* quote: closing it is then the
        model's own natural continuation.

        Tokens are accumulated as raw bytes rather than text, because a
        byte-level BPE token can carry a fragment of a multi-byte UTF-8
        character; decoding only once the value is complete keeps
        multi-byte characters intact.

        Args:
            prompt_ids: Token ids of the prompt the value is conditioned on.
            max_tokens: Hard safety cap on the number of generated tokens.

        Returns:
            The generated string content (without surrounding quotes).
        """
        allowed_ids = self._vocab.string_content_ids
        competitor_ids = _as_id_array(
            allowed_ids | self._vocab.ids_starting_with('"')
        )

        running_ids = list(prompt_ids)
        content = bytearray()
        for _ in range(max_tokens):
            logits = np.asarray(
                self._llm.get_logits_from_input_ids(running_ids),
                dtype=np.float64,
            )
            best_id = _argmax_among(logits, competitor_ids)
            if best_id not in allowed_ids:
                break
            running_ids.append(best_id)
            content += self._vocab.get_bytes(best_id)

        return content.decode("utf-8", errors="replace")
