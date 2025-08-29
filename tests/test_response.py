import pro_engine


def test_response_structure_and_capitalization():
    engine = pro_engine.ProEngine()
    sentence = engine.respond(["hello", "WORLD"])
    words = sentence.rstrip(".").split()
    assert len(words) == 5
    assert words[0][0].isupper()
    assert sentence.endswith(".")
    assert "WORLD" in words[1:]


def test_preserves_first_word_capitalization():
    engine = pro_engine.ProEngine()
    sentence = engine.respond(["NASA", "launch", "window", "opens", "today"])
    words = sentence.rstrip(".").split()
    assert words[0] == "NASA"


def test_fills_from_state_without_duplicates():
    engine = pro_engine.ProEngine()
    engine.state["word_counts"] = {
        "alpha": 5,
        "beta": 4,
        "gamma": 3,
        "delta": 2,
    }
    sentence = engine.respond(["hello"])
    words = sentence.rstrip(".").split()
    assert words == ["Hello", "alpha", "beta", "gamma", "delta"]
