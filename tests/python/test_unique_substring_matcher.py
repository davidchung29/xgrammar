"""Tests for the runtime-bound unique-substring matcher prototype."""

import xgrammar as xgr
from xgrammar.testing import _get_masked_tokens_from_bitmask


def _occurrences(source: bytes, candidate: bytes) -> int:
    if not candidate:
        return len(source) + 1
    return sum(source.startswith(candidate, index) for index in range(len(source)))


def test_runtime_unique_substring_overlapping_occurrences_and_stop():
    vocab = [b"n", b"a", b"na", b"nan", b"x", b"<eos>"]
    tokenizer = xgr.TokenizerInfo(vocab, stop_token_ids=[5])
    matcher = xgr.UniqueSubstringMatcher(b"banana", tokenizer)

    assert matcher.occurrence_count == 7
    assert not matcher.is_completed
    assert matcher.accept_string(b"na")
    assert matcher.occurrence_count == 2
    assert not matcher.accept_token(5)
    assert matcher.accept_string(b"n")
    assert matcher.occurrence_count == 1
    assert matcher.is_completed
    assert matcher.accept_token(5)
    assert matcher.is_terminated

    matcher.rollback(1)
    assert not matcher.is_terminated
    matcher.rollback(2)
    assert matcher.occurrence_count == 7


def test_runtime_unique_substring_mask_matches_reference():
    source = b"aaaa"
    vocab = [b"a", b"aa", b"aaa", b"aaaa", b"aaaaa", b"b", b"<eos>"]
    eos = len(vocab) - 1
    tokenizer = xgr.TokenizerInfo(vocab, stop_token_ids=[eos])
    matcher = xgr.UniqueSubstringMatcher(source, tokenizer)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer.vocab_size)

    prefix = b""
    for _ in range(5):
        matcher.fill_next_token_bitmask(bitmask)
        rejected = set(_get_masked_tokens_from_bitmask(bitmask, tokenizer.vocab_size))
        for token_id, piece in enumerate(vocab[:-1]):
            expected = _occurrences(source, prefix + piece) > 0
            assert (token_id not in rejected) == expected, (prefix, piece)
        assert (eos not in rejected) == (bool(prefix) and _occurrences(source, prefix) == 1)
        if len(prefix) == len(source):
            break
        assert matcher.accept_token(0)
        prefix += b"a"


def test_runtime_unique_substring_reset_and_index_size():
    source = b"def f():\n    return 1\ndef g():\n    return 1\n"
    tokenizer = xgr.TokenizerInfo([b"return 1", b"def f", b"<eos>"], stop_token_ids=[2])
    matcher = xgr.UniqueSubstringMatcher(source, tokenizer)

    assert matcher.num_index_states <= 2 * len(source)
    assert matcher.memory_size_bytes > len(source)
    assert matcher.accept_token(1)
    assert matcher.is_completed
    matcher.reset()
    assert not matcher.is_completed
    assert matcher.occurrence_count == len(source) + 1


def test_runtime_unique_substring_decodes_json_escapes_across_tokens():
    source = 'line 1\n"quoted" \\ path 😀'.encode()
    vocab = [
        b"line 1",
        b"\\",
        b'n\\"quoted\\" ' + b"\\",
        b"\\ path ",
        b"\\uD83D",
        b"\\uDE00",
        b'"',
        b"<eos>",
    ]
    eos = len(vocab) - 1
    tokenizer = xgr.TokenizerInfo(vocab, stop_token_ids=[eos])
    matcher = xgr.UniqueSubstringMatcher(source, tokenizer)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer.vocab_size)

    # The newline and surrogate pair escapes are deliberately divided at token boundaries.
    for token_id in range(6):
        matcher.fill_next_token_bitmask(bitmask)
        assert token_id not in _get_masked_tokens_from_bitmask(bitmask, tokenizer.vocab_size)
        assert matcher.accept_token(token_id)
    assert matcher.is_completed
    assert matcher.accept_token(eos)

    matcher.reset()
    assert not matcher.accept_token(6)  # An unescaped quote closes a JSON string.
    assert not matcher.accept_string(b"\\x")


def test_runtime_unique_substring_rejects_incomplete_json_escape_as_completion():
    tokenizer = xgr.TokenizerInfo([b"a\\", b"n", b"<eos>"], stop_token_ids=[2])
    matcher = xgr.UniqueSubstringMatcher(b"a\n", tokenizer)

    assert matcher.accept_token(0)
    assert not matcher.is_completed
    assert not matcher.accept_token(2)
    assert matcher.accept_token(1)
    assert matcher.is_completed
