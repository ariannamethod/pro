import pro_engine


def test_no_three_single_letters_in_a_row():
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {("a", "b"): {"c": 5, "word": 4}}
    engine.state["word_counts"] = {"a": 5, "b": 5, "c": 5, "word": 1}
    engine.state["bigram_counts"] = {}
    engine.state["char_ngram_counts"] = {}
    engine.state["trigram_inv"] = {}
    engine.state["bigram_inv"] = {}
    engine.state["word_inv"] = {}
    result = engine.plan_sentence(["a", "b"], 3)
    assert result == ["a", "b", "word"]


def test_sentence_does_not_end_with_two_single_letters():
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {("<s>", "a"): {"b": 5, "word": 4}}
    engine.state["word_counts"] = {"a": 5, "b": 5, "word": 1}
    engine.state["bigram_counts"] = {}
    engine.state["char_ngram_counts"] = {}
    engine.state["trigram_inv"] = {}
    engine.state["bigram_inv"] = {}
    engine.state["word_inv"] = {}
    result = engine.plan_sentence(["a"], 2)
    assert result == ["a", "word"]


def test_sentence_can_end_with_single_letter_after_longer_word():
    engine = pro_engine.ProEngine()
    engine.state["trigram_counts"] = {("<s>", "word"): {"a": 5}}
    engine.state["word_counts"] = {"word": 5, "a": 5}
    engine.state["bigram_counts"] = {}
    engine.state["char_ngram_counts"] = {}
    engine.state["trigram_inv"] = {}
    engine.state["bigram_inv"] = {}
    engine.state["word_inv"] = {}
    result = engine.plan_sentence(["word"], 2)
    assert result == ["word", "a"]
