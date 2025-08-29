from pro_identity import swap_pronouns


def test_swap_pronouns_basic():
    assert swap_pronouns(["you", "see", "me"]) == ["I", "see", "me"]
