"""Lightweight linguistic sentence complexity estimation.

The default heuristic estimator is intentionally self-contained so offline
smoke tests do not require external NLP models. The counts are approximations,
not a replacement for a gold syntactic parse.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Set


@dataclass(frozen=True)
class ComplexityFeatures:
    """Sentence complexity features and normalized scalar score."""

    token_count: int
    verb_count: int
    verb_argument_count: int
    noun_count: int
    adjective_count: int
    adverb_count: int
    preposition_count: int
    conjunction_count: int
    subordinate_marker_count: int
    named_entity_count: int
    punctuation_count: int
    raw_score: float
    normalized_score: float
    complexity_1_to_4: float


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
PUNCTUATION_RE = re.compile(r"[,;:()?]")

AUXILIARIES: Set[str] = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "will",
    "would",
    "should",
    "may",
    "might",
    "must",
}
PREPOSITIONS: Set[str] = {
    "in",
    "on",
    "at",
    "by",
    "from",
    "to",
    "with",
    "without",
    "before",
    "after",
    "under",
    "over",
    "between",
    "into",
    "through",
    "around",
    "near",
    "for",
    "of",
    "about",
}
CONJUNCTIONS: Set[str] = {"and", "or", "but"}
SUBORDINATE_MARKERS: Set[str] = {
    "if",
    "unless",
    "while",
    "because",
    "although",
    "before",
    "after",
    "when",
    "whenever",
    "where",
    "whereas",
    "that",
    "which",
    "who",
    "whom",
    "whose",
}
COMPLEMENT_MARKERS: Set[str] = {"to", "that", "whether", "if"}
WH_COMPLEMENTS: Set[str] = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how"}
DETERMINERS: Set[str] = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "our"}
_SPACY_NLP: Optional[Any] = None


def compute_linguistic_complexity(text: str, mode: str = "heuristic") -> ComplexityFeatures:
    """Estimate linguistic complexity for a sentence.

    Args:
        text: Input sentence or utterance.
        mode: ``"heuristic"`` for regex/lexical rules, or ``"spacy"`` to use
            spaCy when installed. spaCy is optional and falls back to the
            heuristic estimator if unavailable.

    Returns:
        ComplexityFeatures with ``normalized_score`` in ``[0, 1]`` and
        ``complexity_1_to_4 = 1 + 3 * normalized_score``.
    """
    if mode == "heuristic":
        return _compute_heuristic_complexity(text)
    if mode == "spacy":
        return _compute_spacy_complexity_or_fallback(text)
    raise ValueError("complexity mode must be one of: heuristic, spacy")


def _compute_heuristic_complexity(text: str) -> ComplexityFeatures:
    original_tokens = TOKEN_RE.findall(text)
    tokens = [token.lower() for token in original_tokens]

    verb_flags = [_is_likely_verb(token) for token in tokens]
    adjective_flags = [_is_likely_adjective(token) for token in tokens]
    adverb_flags = [_is_likely_adverb(token) for token in tokens]

    verb_count = sum(verb_flags)
    adjective_count = sum(adjective_flags)
    adverb_count = sum(adverb_flags)
    preposition_count = sum(token in PREPOSITIONS for token in tokens)
    conjunction_count = sum(token in CONJUNCTIONS or token in SUBORDINATE_MARKERS for token in tokens)
    subordinate_marker_count = sum(token in SUBORDINATE_MARKERS for token in tokens)
    named_entity_count = _count_named_entities(original_tokens)
    punctuation_count = len(PUNCTUATION_RE.findall(text))
    noun_count = sum(
        _is_likely_noun(token, is_verb, is_adjective, is_adverb)
        for token, is_verb, is_adjective, is_adverb in zip(tokens, verb_flags, adjective_flags, adverb_flags)
    )
    verb_argument_count = _estimate_verb_arguments(tokens, verb_flags)

    return _build_features(
        token_count=len(tokens),
        verb_count=verb_count,
        verb_argument_count=verb_argument_count,
        noun_count=noun_count,
        adjective_count=adjective_count,
        adverb_count=adverb_count,
        preposition_count=preposition_count,
        conjunction_count=conjunction_count,
        subordinate_marker_count=subordinate_marker_count,
        named_entity_count=named_entity_count,
        punctuation_count=punctuation_count,
    )


def _compute_spacy_complexity_or_fallback(text: str) -> ComplexityFeatures:
    global _SPACY_NLP
    try:
        if _SPACY_NLP is None:
            import spacy  # type: ignore

            _SPACY_NLP = spacy.load("en_core_web_sm")
    except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
        warnings.warn(
            f"spaCy complexity mode is unavailable ({exc}); falling back to heuristic mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _compute_heuristic_complexity(text)

    doc = _SPACY_NLP(text)
    token_count = sum(not token.is_space and not token.is_punct for token in doc)
    verb_count = sum(token.pos_ in {"VERB", "AUX"} for token in doc)
    noun_count = sum(token.pos_ in {"NOUN", "PROPN", "PRON"} for token in doc)
    adjective_count = sum(token.pos_ == "ADJ" for token in doc)
    adverb_count = sum(token.pos_ == "ADV" for token in doc)
    preposition_count = sum(token.dep_ == "prep" or token.pos_ == "ADP" for token in doc)
    conjunction_count = sum(token.pos_ in {"CCONJ", "SCONJ"} for token in doc)
    subordinate_marker_count = sum(token.pos_ == "SCONJ" or token.text.lower() in SUBORDINATE_MARKERS for token in doc)
    named_entity_count = len(doc.ents)
    punctuation_count = sum(token.text in {",", ";", ":", "(", ")", "?"} for token in doc)
    argument_deps = {"nsubj", "obj", "dobj", "iobj", "ccomp", "xcomp", "acomp", "attr", "obl", "prep", "pobj"}
    verb_argument_count = sum(token.dep_ in argument_deps for token in doc)

    return _build_features(
        token_count=token_count,
        verb_count=verb_count,
        verb_argument_count=verb_argument_count,
        noun_count=noun_count,
        adjective_count=adjective_count,
        adverb_count=adverb_count,
        preposition_count=preposition_count,
        conjunction_count=conjunction_count,
        subordinate_marker_count=subordinate_marker_count,
        named_entity_count=named_entity_count,
        punctuation_count=punctuation_count,
    )


def _build_features(
    *,
    token_count: int,
    verb_count: int,
    verb_argument_count: int,
    noun_count: int,
    adjective_count: int,
    adverb_count: int,
    preposition_count: int,
    conjunction_count: int,
    subordinate_marker_count: int,
    named_entity_count: int,
    punctuation_count: int,
) -> ComplexityFeatures:
    raw_score = (
        0.10 * token_count
        + 0.40 * verb_count
        + 0.60 * verb_argument_count
        + 0.25 * adjective_count
        + 0.25 * adverb_count
        + 0.30 * preposition_count
        + 0.35 * conjunction_count
        + 0.45 * subordinate_marker_count
        + 0.20 * named_entity_count
        + 0.10 * punctuation_count
    )
    normalized_score = _clamp(raw_score / (raw_score + 10.0) if raw_score > 0.0 else 0.0)
    return ComplexityFeatures(
        token_count=int(token_count),
        verb_count=int(verb_count),
        verb_argument_count=int(verb_argument_count),
        noun_count=int(noun_count),
        adjective_count=int(adjective_count),
        adverb_count=int(adverb_count),
        preposition_count=int(preposition_count),
        conjunction_count=int(conjunction_count),
        subordinate_marker_count=int(subordinate_marker_count),
        named_entity_count=int(named_entity_count),
        punctuation_count=int(punctuation_count),
        raw_score=float(raw_score),
        normalized_score=float(normalized_score),
        complexity_1_to_4=float(1.0 + 3.0 * normalized_score),
    )


def _is_likely_verb(token: str) -> bool:
    return token in AUXILIARIES or (len(token) > 4 and token.endswith("ed")) or (len(token) > 5 and token.endswith("ing"))


def _is_likely_adjective(token: str) -> bool:
    return token.endswith(("ive", "al", "ous", "ful", "less", "able", "ible", "ic", "ical"))


def _is_likely_adverb(token: str) -> bool:
    return len(token) > 4 and token.endswith("ly")


def _is_likely_noun(token: str, is_verb: bool, is_adjective: bool, is_adverb: bool) -> bool:
    if not token.isalpha() or len(token) <= 2:
        return False
    if token in AUXILIARIES or token in PREPOSITIONS or token in CONJUNCTIONS or token in SUBORDINATE_MARKERS:
        return False
    if token in DETERMINERS or is_verb or is_adjective or is_adverb:
        return False
    return True


def _estimate_verb_arguments(tokens: Sequence[str], verb_flags: Sequence[bool]) -> int:
    """Heuristically count likely verb arguments.

    This is not a gold syntactic parse. It only looks for local lexical cues:
    noun-like tokens after verbs, prepositional complements, complementizers,
    and wh-complement markers.
    """
    argument_count = 0
    for index, is_verb in enumerate(verb_flags):
        if not is_verb:
            continue
        window = list(enumerate(tokens[index + 1 : index + 7], start=index + 1))
        if not window:
            continue

        saw_direct_object = False
        for window_index, token in window:
            if token in CONJUNCTIONS:
                break
            if token in COMPLEMENT_MARKERS or token in WH_COMPLEMENTS:
                argument_count += 1
                continue
            if token in PREPOSITIONS:
                following = tokens[window_index + 1 : min(window_index + 4, len(tokens))]
                if any(_nounish_for_argument(candidate) for candidate in following):
                    argument_count += 1
                continue
            if not saw_direct_object and _nounish_for_argument(token):
                argument_count += 1
                saw_direct_object = True
    return argument_count


def _nounish_for_argument(token: str) -> bool:
    if token in DETERMINERS or token in PREPOSITIONS or token in CONJUNCTIONS or token in SUBORDINATE_MARKERS:
        return False
    return token.isalpha() and len(token) > 2 and not _is_likely_adverb(token)


def _count_named_entities(tokens: Sequence[str]) -> int:
    count = 0
    in_span = False
    for index, token in enumerate(tokens):
        is_capitalized_non_initial = index > 0 and token[:1].isupper() and token[1:].islower()
        if is_capitalized_non_initial and not in_span:
            count += 1
            in_span = True
        elif not is_capitalized_non_initial:
            in_span = False
    return count


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
