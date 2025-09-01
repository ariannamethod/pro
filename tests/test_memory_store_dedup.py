from memory import MemoryStore


def test_add_utterance_skips_duplicates():
    store = MemoryStore()
    assert store.add_utterance("dlg", "user", "hello") is not None
    assert store.add_utterance("dlg", "user", "hello") is None
    assert len(store.get_dialogue("dlg")) == 1

