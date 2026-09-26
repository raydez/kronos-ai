import pytest

from kronos_ai.domain.symbols import is_normalized_symbol, normalize_symbol


class TestNormalizeSymbol:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("600000", "600000"),
            ("sh.600000", "600000"),
            ("SZ.000001", "000001"),
            ("bj.830799", "830799"),
            (" 000001 ", "000001"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_symbol(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["", "60000", "6000000", "abc.def", "sh.60000", "hk.00700", "sh600000"],
    )
    def test_invalid_rejected(self, raw: str) -> None:
        with pytest.raises(ValueError, match="symbol"):
            normalize_symbol(raw)

    def test_is_normalized(self) -> None:
        assert is_normalized_symbol("600000")
        assert not is_normalized_symbol("sh.600000")
