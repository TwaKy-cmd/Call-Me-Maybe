"""Tests for src.vocabulary: the precomputed JSON-grammar masks."""

import json

import pytest

from src.vocabulary import NumberState, Vocabulary


def test_missing_file_raises_oserror(tmp_path) -> None:
    with pytest.raises(OSError):
        Vocabulary(str(tmp_path / "does_not_exist.json"))


def test_invalid_json_raises_value_error(tmp_path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text("{not valid json")
    with pytest.raises(ValueError):
        Vocabulary(str(path))


def test_non_object_json_raises_value_error(tmp_path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(["a", "b"]))
    with pytest.raises(ValueError):
        Vocabulary(str(path))


def test_get_text_and_get_id_roundtrip(vocabulary: Vocabulary) -> None:
    token_id = vocabulary.get_id("a")
    assert token_id is not None
    assert vocabulary.get_text(token_id) == "a"
    assert vocabulary.get_id("does-not-exist-as-a-token") is None


@pytest.fixture
def byte_level_vocabulary(tmp_path) -> Vocabulary:
    """A vocabulary written in the byte-level BPE alphabet, like Qwen's."""
    path = tmp_path / "vocab.json"
    path.write_text(
        json.dumps(
            {
                "Ġshrek": 0,  # " shrek"
                "Ċ": 1,  # "\n"
                "hello": 2,
                '"': 3,
                "Ġ": 4,  # " "
                "âĤ¬": 5,  # "€", split across three raw bytes
            }
        )
    )
    return Vocabulary(str(path))


def test_byte_level_tokens_decode_to_real_text(byte_level_vocabulary: Vocabulary) -> None:
    assert byte_level_vocabulary.get_text(0) == " shrek"
    assert byte_level_vocabulary.get_text(1) == "\n"
    assert byte_level_vocabulary.get_id(" shrek") == 0
    assert byte_level_vocabulary.get_id("\n") == 1


def test_byte_level_multibyte_character_survives_concatenation(
    byte_level_vocabulary: Vocabulary,
) -> None:
    joined = b"".join(byte_level_vocabulary.get_bytes(i) for i in (2, 5))
    assert joined.decode("utf-8") == "hello€"


def test_byte_level_newline_is_not_valid_string_content(
    byte_level_vocabulary: Vocabulary,
) -> None:
    content_ids = byte_level_vocabulary.string_content_ids
    assert 1 not in content_ids  # "\n" is a control byte
    assert 3 not in content_ids  # '"' would close the string
    assert 0 in content_ids
    assert 4 in content_ids


def test_string_content_ids_excludes_forbidden_characters(vocabulary: Vocabulary) -> None:
    quote_id = vocabulary.get_id('"')
    letter_id = vocabulary.get_id("a")
    assert quote_id not in vocabulary.string_content_ids
    assert letter_id in vocabulary.string_content_ids


def test_number_mask_start_allows_digit_and_minus(vocabulary: Vocabulary) -> None:
    allowed, end_states = vocabulary.number_mask(NumberState.START)
    digit_id = vocabulary.get_id("4")
    minus_id = vocabulary.get_id("-")
    letter_id = vocabulary.get_id("a")

    assert digit_id in allowed
    assert end_states[digit_id] is NumberState.INT_DIGITS
    assert minus_id in allowed
    assert end_states[minus_id] is NumberState.AFTER_SIGN
    assert letter_id not in allowed


def test_number_mask_after_sign_only_allows_digit(vocabulary: Vocabulary) -> None:
    allowed, end_states = vocabulary.number_mask(NumberState.AFTER_SIGN)
    assert vocabulary.get_id("-") not in allowed
    digit_id = vocabulary.get_id("7")
    assert digit_id in allowed
    assert end_states[digit_id] is NumberState.INT_DIGITS


def test_number_mask_int_digits_allows_digit_and_dot(vocabulary: Vocabulary) -> None:
    allowed, end_states = vocabulary.number_mask(NumberState.INT_DIGITS)
    dot_id = vocabulary.get_id(".")
    digit_id = vocabulary.get_id("9")

    assert dot_id in allowed
    assert end_states[dot_id] is NumberState.AFTER_DOT
    assert digit_id in allowed
    assert end_states[digit_id] is NumberState.INT_DIGITS
    assert vocabulary.get_id("-") not in allowed


def test_number_mask_after_dot_requires_digit(vocabulary: Vocabulary) -> None:
    allowed, end_states = vocabulary.number_mask(NumberState.AFTER_DOT)
    assert vocabulary.get_id(".") not in allowed
    digit_id = vocabulary.get_id("1")
    assert digit_id in allowed
    assert end_states[digit_id] is NumberState.FRAC_DIGITS


def test_is_accepting_number_state() -> None:
    assert Vocabulary.is_accepting_number_state(NumberState.INT_DIGITS)
    assert Vocabulary.is_accepting_number_state(NumberState.FRAC_DIGITS)
    assert not Vocabulary.is_accepting_number_state(NumberState.START)
    assert not Vocabulary.is_accepting_number_state(NumberState.AFTER_SIGN)
    assert not Vocabulary.is_accepting_number_state(NumberState.AFTER_DOT)
