"""Shared pytest fixtures: a tiny character-level vocabulary and a fake LLM.

Running the real Qwen3-0.6B model is out of scope for this fast unit-test
suite (see the README's Testing strategy section for how the pipeline was
validated against the real model). Instead, ``OracleLLM`` below exercises
the exact same code paths (``get_logits_from_input_ids``, ``encode``) as
the real SDK, letting every constrained-decoding decision be driven by a
small, fully deterministic fake model.
"""

import json
from typing import Iterable

import pytest

from src.vocabulary import Vocabulary

#: A character-level vocabulary (one token per printable ASCII character,
#: plus newline) -- enough to tokenize any prompt text the Generator can
#: build, and every code path, without needing the real BPE tokenizer.
_CHARACTERS = [chr(code) for code in range(32, 127)] + ["\n"]
TEST_VOCAB = {char: token_id for token_id, char in enumerate(_CHARACTERS)}


@pytest.fixture
def vocab_path(tmp_path) -> str:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(TEST_VOCAB))
    return str(path)


@pytest.fixture
def vocabulary(vocab_path: str) -> Vocabulary:
    return Vocabulary(vocab_path)


class _FakeTensor:
    """Minimal stand-in for the torch.Tensor returned by encode()."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def flatten(self) -> "_FakeTensor":
        return self

    def tolist(self) -> list[int]:
        return self._ids


class OracleLLM:
    """Fake LLM whose logits chase a hidden per-decision target string.

    It decodes ``input_ids`` back to text (this test vocabulary is
    one-character-per-token, so decoding is exact), finds which
    registered prompt marker the text currently ends with, and boosts
    whichever token continues that marker's target string. Once the
    target has been fully produced, it boosts the configured stop ids
    instead -- exercising the real "keep going vs stop" competition
    implemented in :mod:`src.constraint_engine`.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        targets: dict[str, str],
        stop_ids: Iterable[int],
    ):
        self._vocab = vocabulary
        self._targets = targets
        self._stop_ids = set(stop_ids)
        self._size = max(vocabulary.text_to_id.values()) + 1

    def encode(self, text: str) -> _FakeTensor:
        return _FakeTensor([self._vocab.text_to_id[char] for char in text])

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        text = "".join(
            self._vocab.get_text(token_id) for token_id in input_ids
        )
        logits = [0.0] * self._size

        marker, generated = self._active_marker(text)
        if marker is None:
            return logits
        target = self._targets[marker]

        if generated == target:
            for token_id in self._stop_ids:
                logits[token_id] = 5.0
            return logits

        if target.startswith(generated):
            next_char = target[len(generated)]
            logits[self._vocab.text_to_id[next_char]] = 10.0
        return logits

    def _active_marker(self, text: str) -> tuple[str | None, str]:
        best_marker = None
        best_pos = -1
        for marker in self._targets:
            pos = text.rfind(marker)
            if pos > best_pos:
                best_pos = pos
                best_marker = marker
        if best_marker is None:
            return None, ""
        return best_marker, text[best_pos + len(best_marker):]
