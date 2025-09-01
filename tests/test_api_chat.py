import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.chat import ChatAPI  # noqa: E402


def test_process_message_returns_vector_and_context():
    api = ChatAPI()
    vec1, ctx1 = api.process_message("dlg", "user", "hello")
    assert isinstance(vec1, np.ndarray)
    assert ctx1["dialogue_id"] == "dlg"
    assert ctx1["speaker"] == "user"
    assert ctx1["messages"] == ["hello"]

    vec2, ctx2 = api.process_message("dlg", "bot", "hi there")
    assert ctx2["messages"] == ["hello", "hi there"]
    assert api.retriever.last_message("dlg", "bot") == "hi there"
