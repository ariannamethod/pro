from typing import List

PRONOUN_MAP = {
    "you": "I",
    "your": "my",
    "yours": "mine",
    "yourself": "myself",
    "yourselves": "ourselves",
}


def swap_pronouns(tokens: List[str]) -> List[str]:
    """Swap first and second person pronouns using PRONOUN_MAP."""
    return [PRONOUN_MAP.get(tok, tok) for tok in tokens]
