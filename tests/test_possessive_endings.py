import asyncio

import pro_engine
import pro_predict


def test_sentences_do_not_end_with_possessive_pronouns(monkeypatch):
    engine = pro_engine.ProEngine()
    engine.state["word_counts"] = {"alpha": 5, "beta": 4, "gamma": 3}
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
        count = getattr(self, "_calls", 0) + 1
        self._calls = count
        if count == 1:
            return ["alpha"] * (target_length - 1) + ["my"]
        return ["beta"] * (target_length - 1) + ["their"]

    monkeypatch.setattr(pro_engine.ProEngine, "plan_sentence", fake_plan_sentence, raising=False)

    sentence = asyncio.run(engine.respond(["seed"]))
    first, second = sentence.split(". ")
    assert first.split()[-1].lower() not in {"his", "my", "their"}
    assert second.rstrip(".").split()[-1].lower() not in {"his", "my", "their"}
