import pro_predict


def test_combination_uses_transformer_when_no_ngram():
    logits = {"world": 0.9, "hello": 0.1}
    res = pro_predict.combine_predictions("", logits)
    assert res[0] == "world"
    assert len(res) == len(set(res))


def test_combination_uses_ngram_when_no_transformer():
    res = pro_predict.combine_predictions("hello", {})
    assert res == ["hello"]


def test_combination_no_duplicate_words():
    logits = {"hello": 0.9, "world": 0.1}
    res = pro_predict.combine_predictions("hello", logits)
    assert res[0] == "hello"
    assert len(res) == len(set(res))
