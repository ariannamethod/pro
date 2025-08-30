import grammar_filters


def test_capital_the_in_middle_is_rejected():
    assert not grammar_filters.passes_filters("We saw The cat")


def test_article_before_pronoun_is_rejected():
    assert not grammar_filters.passes_filters("a she walked")
    assert not grammar_filters.passes_filters("the it moved")


def test_capital_the_at_start_is_allowed():
    assert grammar_filters.passes_filters("The dog ran")
