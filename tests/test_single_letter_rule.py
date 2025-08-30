import asyncio

import pro_engine
import pro_predict


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


def test_second_sentence_respects_single_letter_rule(monkeypatch):
    engine = pro_engine.ProEngine()
    engine.state["word_counts"] = {"x": 5, "a": 4, "b": 3, "c": 2, "d": 1}
    engine.state["bigram_counts"] = {}
    engine.state["trigram_counts"] = {}
    engine.state["char_ngram_counts"] = {}
    engine.state["trigram_inv"] = {}
    engine.state["bigram_inv"] = {}
    engine.state["word_inv"] = {}

    monkeypatch.setattr(pro_predict, "lookup_analogs", lambda w: w)
    monkeypatch.setattr(pro_predict, "_ensure_vectors", lambda: None)
    pro_predict._VECTORS = {w: {w: 1.0} for w in engine.state["word_counts"]}

    def fake_plan_sentence(self, initial, target_length, **kwargs):
        if len(initial) > 2:
            return initial[:target_length]
        return ["word"] * (target_length - 1) + ["a"]

    monkeypatch.setattr(pro_engine.ProEngine, "plan_sentence", fake_plan_sentence, raising=False)

    sentence = asyncio.run(engine.respond(["x"]))
    _, second = sentence.split(". ")
    last_words = second.rstrip(".").split()
    assert not (len(last_words[-1]) == len(last_words[-2]) == 1)
