from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are a computational linguist and philologist specializing in Classical Tibetan texts.\n\n"
    "You will be given a sentence written in Tibetan extended Wylie transliteration, and a similarity level between 0.0 and 1.0.\n\n"
    "Your task is to produce a new Wylie sentence whose meaning is controlled by that similarity level.\n\n"
    "Similarity scale:\n"
    "- 1.0 = almost identical meaning; very close paraphrase\n"
    "- 0.8 = strong semantic similarity; same idea with noticeable variation\n"
    "- 0.5 = moderate similarity; related idea within the same topic\n"
    "- 0.2 = weak similarity; distant but still in the same thematic domain\n"
    "- 0.0 = minimal similarity; only loosely related topic\n\n"
    "Constraints:\n"
    "- The sentence must remain relevant in meaning to the topic\n"
    "- remain in the same semantic domain and register\n"
    "- avoid copying structure or word order\n"
    "- introduce lexical variation where possible\n"
    "- sound like a plausible Classical Tibetan sentence\n"
    "- do not introduce unrelated topics\n\n"
    "You may paraphrase, summarize, abstract, or elaborate according to the requested similarity.\n"
    "Return only the generated Wylie sentence."
)

HUMAN_PROMPT_TEMPLATE = (
    "Original sentence (Wylie):\n"
    "{sentence}\n\n"
    "Target similarity level (0–1): {similarity}\n\n"
    "Write a Tibetan sentence in Wylie with that similarity:\n"
)


def build_messages(sentence: str, similarity: str) -> list:
    """Build LangChain messages for the given sentence."""
    msg = HUMAN_PROMPT_TEMPLATE.format(sentence=sentence.strip(), similarity=similarity)
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=msg)]
