"""Small demonstration of the memory graph store and retriever."""

from api.chat import ChatAPI


def main() -> None:
    chat = ChatAPI(path="demo_memory.json")
    dialogue = "demo"
    chat.process_message(dialogue, "user", "My name is Alice")
    # The model will remember the last user message when processing another
    # one.  MemoryAttention encodes the remembered text into the hidden state.
    vec = chat.process_message(dialogue, "bot", "What is my name?")
    print("Hidden state influenced by memory:", vec)


if __name__ == "__main__":
    main()
