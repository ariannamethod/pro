import pytest
import grammar_filters


@pytest.mark.parametrize("word", ["of", "from", "where", "when"])
def test_final_preposition_casing(word):
    assert grammar_filters.passes_filters(f"This ends with {word}.")
    assert not grammar_filters.passes_filters(
        f"This ends with {word.capitalize()}."
    )
    assert grammar_filters.passes_filters(f"This ends with {word.upper()}.")
