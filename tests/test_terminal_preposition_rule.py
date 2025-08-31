import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import grammar_filters  # noqa: E402


def test_lowercase_terminal_preposition_allowed():
    assert grammar_filters.passes_filters("We looked at.")


def test_mixed_case_terminal_preposition_disallowed():
    assert not grammar_filters.passes_filters("We looked At.")


def test_uppercase_terminal_preposition_allowed():
    assert grammar_filters.passes_filters("We looked AT.")
