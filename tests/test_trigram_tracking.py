import pro_sequence
import pro_metrics


def test_trigram_and_char_ngrams():
    state = {}
    words = ["one", "two", "three"]
    pro_sequence.analyze_sequences(state, words)
    assert state["trigram_counts"][('<s>', '<s>')]["one"] == 1
    assert state["trigram_counts"][('<s>', 'one')]["two"] == 1
    assert state["trigram_counts"][('one', 'two')]["three"] == 1
    assert state["char_ngram_counts"]["one"] == 1
    metrics = pro_metrics.compute_metrics(
        words,
        state["trigram_counts"],
        state["bigram_counts"],
        state["word_counts"],
        state["char_ngram_counts"],
    )
    assert metrics["trigram_resonance"] > 0
    assert metrics["char_ngram_resonance"] > 0
