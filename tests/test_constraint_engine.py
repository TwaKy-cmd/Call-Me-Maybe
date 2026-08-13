"""Tests for src.constraint_engine: the core constrained-decoding logic."""

import pytest

from src.constraint_engine import ChoiceDecoder, DecodingError, ValueDecoder
from tests.conftest import OracleLLM


def test_choice_decoder_picks_the_scripted_candidate(vocabulary) -> None:
    llm = OracleLLM(vocabulary, targets={"pick:": "dog"}, stop_ids=set())
    decoder = ChoiceDecoder(llm)
    prompt_ids = llm.encode("pick:").flatten().tolist()

    result = decoder.choose(prompt_ids, ["cat", "dog"])

    assert result == "dog"


def test_choice_decoder_single_candidate_needs_no_llm_call(vocabulary) -> None:
    class ExplodingLLM:
        def get_logits_from_input_ids(self, input_ids):
            raise AssertionError("should not be called when there is only one candidate")

        def encode(self, text):
            raise AssertionError("should not be called when there is only one candidate")

    decoder = ChoiceDecoder(ExplodingLLM())
    assert decoder.choose([], ["only_option"]) == "only_option"


def test_choice_decoder_stops_at_a_shared_prefix_when_stop_wins(vocabulary) -> None:
    # "hi" is itself a valid candidate but also a prefix of "hidden".
    quote_id = vocabulary.get_id('"')
    llm = OracleLLM(vocabulary, targets={"pick:": "hi"}, stop_ids={quote_id})
    decoder = ChoiceDecoder(llm)
    prompt_ids = llm.encode("pick:").flatten().tolist()

    result = decoder.choose(prompt_ids, ["hi", "hidden"], stop_ids=frozenset({quote_id}))

    assert result == "hi"


def test_choice_decoder_continues_past_shared_prefix_when_no_stop_signal(vocabulary) -> None:
    llm = OracleLLM(vocabulary, targets={"pick:": "hidden"}, stop_ids=set())
    decoder = ChoiceDecoder(llm)
    prompt_ids = llm.encode("pick:").flatten().tolist()

    result = decoder.choose(prompt_ids, ["hi", "hidden"], stop_ids=frozenset())

    assert result == "hidden"


def test_choice_decoder_requires_at_least_one_candidate() -> None:
    class UnusedLLM:
        def get_logits_from_input_ids(self, input_ids):
            return []

        def encode(self, text):
            raise AssertionError

    with pytest.raises(ValueError):
        ChoiceDecoder(UnusedLLM()).choose([], [])


def test_generate_number_produces_an_integer(vocabulary) -> None:
    stop_ids = {vocabulary.get_id(","), vocabulary.get_id("}")}
    llm = OracleLLM(vocabulary, targets={"a =": "42"}, stop_ids=stop_ids)
    decoder = ValueDecoder(llm, vocabulary)
    prompt_ids = llm.encode("a =").flatten().tolist()

    result = decoder.generate_number(prompt_ids, integer=True, stop_ids=frozenset(stop_ids))

    assert result == 42
    assert isinstance(result, int)


def test_generate_number_produces_a_float(vocabulary) -> None:
    stop_ids = {vocabulary.get_id(","), vocabulary.get_id("}")}
    llm = OracleLLM(vocabulary, targets={"a =": "4.2"}, stop_ids=stop_ids)
    decoder = ValueDecoder(llm, vocabulary)
    prompt_ids = llm.encode("a =").flatten().tolist()

    result = decoder.generate_number(prompt_ids, integer=False, stop_ids=frozenset(stop_ids))

    assert result == pytest.approx(4.2)
    assert isinstance(result, float)


def test_generate_number_handles_negative_values(vocabulary) -> None:
    stop_ids = {vocabulary.get_id(","), vocabulary.get_id("}")}
    llm = OracleLLM(vocabulary, targets={"a =": "-7"}, stop_ids=stop_ids)
    decoder = ValueDecoder(llm, vocabulary)
    prompt_ids = llm.encode("a =").flatten().tolist()

    result = decoder.generate_number(prompt_ids, integer=True, stop_ids=frozenset(stop_ids))

    assert result == -7


def test_generate_string_stops_at_closing_quote(vocabulary) -> None:
    quote_id = vocabulary.get_id('"')
    llm = OracleLLM(vocabulary, targets={"name =": "bob"}, stop_ids={quote_id})
    decoder = ValueDecoder(llm, vocabulary)
    prompt_ids = llm.encode("name =").flatten().tolist()

    result = decoder.generate_string(prompt_ids)

    assert result == "bob"


def test_generate_string_can_produce_an_empty_string(vocabulary) -> None:
    quote_id = vocabulary.get_id('"')
    llm = OracleLLM(vocabulary, targets={"name =": ""}, stop_ids={quote_id})
    decoder = ValueDecoder(llm, vocabulary)
    prompt_ids = llm.encode("name =").flatten().tolist()

    result = decoder.generate_string(prompt_ids)

    assert result == ""


def test_generate_number_raises_when_nothing_is_representable(vocabulary) -> None:
    class BlankLLM:
        def get_logits_from_input_ids(self, input_ids):
            return [float("-inf")] * (max(vocabulary.text_to_id.values()) + 1)

        def encode(self, text):
            raise AssertionError

    decoder = ValueDecoder(BlankLLM(), vocabulary)
    with pytest.raises(DecodingError):
        decoder.generate_number([], integer=False, stop_ids=frozenset())
