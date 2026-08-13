"""Token vocabulary loading and precomputed JSON-grammar masks.

The vocabulary file maps each token's literal text to its integer id, but
that text is *not* the text the token stands for: byte-level BPE
tokenizers (GPT-2, Qwen...) re-encode every raw byte into a printable
placeholder character, so a space is stored as ``Ġ`` and a newline as
``Ċ``. :func:`_byte_decoder` rebuilds the inverse table once at import
time, and every token is turned back into the raw ``bytes`` it really
represents. Working in bytes rather than ``str`` matters because a single
token may hold only *part* of a multi-byte UTF-8 character; the decoders
accumulate bytes and decode once the value is complete.

Every mask below is computed a single time in ``__init__`` so that
per-token generation steps only need cheap set lookups, keeping the
pipeline fast even with a ~150k-token vocabulary.
"""

import json
from enum import Enum, auto

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

#: Validates the raw vocabulary file: a mapping of token text to token id.
_VOCAB_ADAPTER: TypeAdapter[dict[str, int]] = TypeAdapter(dict[str, int])

#: Bytes that may never appear raw inside a JSON string: the control
#: range, the closing quote, and the escape character.
_QUOTE_BYTE = 0x22
_BACKSLASH_BYTE = 0x5C
_FIRST_PRINTABLE_BYTE = 0x20


class NumberState(Enum):
    """States of the small DFA describing a syntactically valid JSON number."""

    START = auto()
    AFTER_SIGN = auto()
    INT_DIGITS = auto()
    AFTER_DOT = auto()
    FRAC_DIGITS = auto()


_ACCEPTING_NUMBER_STATES = frozenset({NumberState.INT_DIGITS, NumberState.FRAC_DIGITS})


def _byte_decoder() -> dict[str, int]:
    """Build the byte-level BPE alphabet, inverted: placeholder char -> byte.

    This mirrors the ``bytes_to_unicode`` table shared by every GPT-2
    style byte-level BPE tokenizer: printable ASCII and Latin-1 ranges map
    to themselves, and the remaining 68 bytes (control characters, space,
    ...) are shifted into the unused U+0100.. range so that a vocabulary
    file stays plain printable text.

    Returns:
        A mapping from placeholder character to the byte value it encodes.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    placeholders = printable[:]
    shifted = 0
    for byte in range(2**8):
        if byte not in printable:
            printable.append(byte)
            placeholders.append(2**8 + shifted)
            shifted += 1
    return {chr(code): byte for byte, code in zip(printable, placeholders)}


_BYTE_DECODER = _byte_decoder()


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

        try:
            data = _VOCAB_ADAPTER.validate_python(data)
        except PydanticValidationError as e:
            raise ValueError(
                f"Vocabulary file '{vocab_path}' must map token strings to integer ids: {e}"
            ) from e
        if not data:
            raise ValueError(f"Vocabulary file '{vocab_path}' must contain a non-empty object")

        #: Raw file contents, i.e. still in the byte-level BPE alphabet.
        self.text_to_id: dict[str, int] = data
        self.id_to_text: dict[int, str] = {v: k for k, v in data.items()}

        #: The bytes each token actually stands for, once the byte-level
        #: BPE placeholders have been resolved.
        self.id_to_bytes: dict[int, bytes] = {
            token_id: self._decode_token(text) for text, token_id in data.items()
        }
        self._bytes_to_id: dict[bytes, int] = {}
        for token_id, raw in self.id_to_bytes.items():
            self._bytes_to_id.setdefault(raw, token_id)

        self.string_content_ids: frozenset[int] = self._build_string_content_ids()
        self._number_masks: dict[NumberState, NumberMask] = self._build_number_masks()
        self._prefix_cache: dict[bytes, frozenset[int]] = {}

    def ids_starting_with(self, text: str) -> frozenset[int]:
        """Ids of every token whose decoded text *starts with* `text`.

        A byte-level BPE tokenizer merges a closing delimiter with what
        follows it, so the token that really ends a value is rarely the
        bare delimiter: closing a JSON string mid-object is one `","`
        token, not `"` then `,`. Looking only for the bare delimiter
        therefore misses the model's actual intention almost every time.
        Every token starting with the delimiter carries the same "this
        value is over" decision, so they all belong in the stop set.
        """
        raw = text.encode("utf-8")
        cached = self._prefix_cache.get(raw)
        if cached is None:
            cached = frozenset(
                token_id for token_id, value in self.id_to_bytes.items() if value.startswith(raw)
            )
            self._prefix_cache[raw] = cached
        return cached

    @staticmethod
    def _decode_token(text: str) -> bytes:
        """Turn one vocabulary key into the raw bytes it encodes.

        Characters outside the byte-level alphabet are kept as their own
        UTF-8 encoding, so plain (non byte-level) vocabularies also load.
        """
        out = bytearray()
        for char in text:
            byte = _BYTE_DECODER.get(char)
            if byte is None:
                out += char.encode("utf-8")
            else:
                out.append(byte)
        return bytes(out)

    def get_bytes(self, token_id: int) -> bytes:
        """Return the raw bytes a token id stands for."""
        return self.id_to_bytes[token_id]

    def get_text(self, token_id: int) -> str:
        """Return the decoded text of a token id.

        Only safe for tokens known to hold whole characters (digits, JSON
        punctuation...). Use :meth:`get_bytes` when concatenating
        arbitrary generated tokens.
        """
        return self.id_to_bytes[token_id].decode("utf-8", errors="replace")

    def get_id(self, text: str) -> int | None:
        """Return the token id whose decoded text is exactly `text`, if any."""
        return self._bytes_to_id.get(text.encode("utf-8"))

    def _build_string_content_ids(self) -> frozenset[int]:
        """Ids of every token usable verbatim inside a JSON string."""
        return frozenset(
            token_id
            for token_id, raw in self.id_to_bytes.items()
            if raw
            and all(
                byte >= _FIRST_PRINTABLE_BYTE and byte not in (_QUOTE_BYTE, _BACKSLASH_BYTE)
                for byte in raw
            )
        )

    def _build_number_masks(self) -> dict[NumberState, NumberMask]:
        masks: dict[NumberState, NumberMask] = {}
        for state in NumberState:
            allowed: set[int] = set()
            end_states: dict[int, NumberState] = {}
            for token_id, raw in self.id_to_bytes.items():
                if not raw:
                    continue
                try:
                    text = raw.decode("ascii")
                except UnicodeDecodeError:
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
