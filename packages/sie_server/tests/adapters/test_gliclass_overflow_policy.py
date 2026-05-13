from __future__ import annotations

import pytest
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.gliclass import GLiClassAdapter


class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[str]]:
        return {"input_ids": text.split()}

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(ids)


class _FakePipe:
    def prepare_input(self, text: str, labels: list[str]) -> str:
        prefix = "<LABEL> " + " <SEP> ".join(labels)
        return f"{prefix} {text}".strip()


class _FakePipeline:
    pipe = _FakePipe()


# With _FakeTokenizer + _FakePipe, N labels produce a 2N-token label_prompt.
# _LABELS → "<LABEL> x <SEP> y" = 4 tokens; with special_count=2 the overhead
# is 6, so for max_seq_length=10 the per-text budget is 4 tokens.
_LABELS = ["x", "y"]


def _make_adapter(*, max_seq_length: int = 10, special_count: int = 2) -> GLiClassAdapter:
    adapter = GLiClassAdapter("test-model")
    adapter._max_seq_length = max_seq_length
    adapter._special_count = special_count
    adapter._tokenizer = _FakeTokenizer()  # ty:ignore[invalid-assignment]
    adapter._pipeline = _FakePipeline()  # ty:ignore[invalid-assignment]
    return adapter


class TestApplyOverflowPolicy:
    def test_default_is_noop_even_when_input_overflows(self) -> None:
        adapter = _make_adapter()
        texts = ["a b c d e f g h i j"]  # 10 tokens, observed = 16 > 10

        assert adapter._apply_overflow_policy(texts, _LABELS, "default") == texts

    def test_default_is_the_default_arg(self) -> None:
        adapter = _make_adapter()
        texts = ["a b c d e f g h i j"]

        assert adapter._apply_overflow_policy(texts, _LABELS) == texts

    def test_truncate_text_passes_fitting_text_through(self) -> None:
        adapter = _make_adapter()
        texts = ["a b c d"]  # 4 tokens, observed = 10

        assert adapter._apply_overflow_policy(texts, _LABELS, "truncate_text") == ["a b c d"]

    def test_truncate_text_slices_overflowing_text_to_budget(self) -> None:
        adapter = _make_adapter()
        texts = ["a b c d e f g"]  # 7 tokens, budget = 4

        assert adapter._apply_overflow_policy(texts, _LABELS, "truncate_text") == ["a b c d"]

    def test_truncate_text_mixed_batch(self) -> None:
        adapter = _make_adapter()
        texts = ["a b", "a b c d e f g"]

        assert adapter._apply_overflow_policy(texts, _LABELS, "truncate_text") == ["a b", "a b c d"]

    def test_error_raises_on_overflowing_text(self) -> None:
        adapter = _make_adapter()
        texts = ["a b c d e"]  # 5 tokens, observed = 11 > 10

        with pytest.raises(InputTooLongError, match=r"items\[0\] observed_tokens=11"):
            adapter._apply_overflow_policy(texts, _LABELS, "error")

    def test_error_reports_first_overflowing_item_index(self) -> None:
        adapter = _make_adapter()
        texts = ["a b", "a b c d e f"]

        with pytest.raises(InputTooLongError, match=r"items\[1\]"):
            adapter._apply_overflow_policy(texts, _LABELS, "error")

    def test_label_prompt_overflow_raises_under_truncate_text(self) -> None:
        adapter = _make_adapter(max_seq_length=5)  # overhead 6 > 5

        with pytest.raises(InputTooLongError, match="label_prompt"):
            adapter._apply_overflow_policy(["a"], _LABELS, "truncate_text")

    def test_label_prompt_overflow_raises_under_error(self) -> None:
        adapter = _make_adapter(max_seq_length=5)

        with pytest.raises(InputTooLongError, match="label_prompt"):
            adapter._apply_overflow_policy(["a"], _LABELS, "error")

    def test_label_prompt_overflow_does_not_raise_under_default(self) -> None:
        adapter = _make_adapter(max_seq_length=5)
        texts = ["a b c"]

        assert adapter._apply_overflow_policy(texts, _LABELS, "default") == texts
