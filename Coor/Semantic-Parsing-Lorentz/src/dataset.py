"""Dummy semantic parsing data and tokenization utilities."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LabelSpec:
    name: str
    templates: Sequence[str]


BASE_LABEL_SPECS: Sequence[LabelSpec] = (
    LabelSpec(
        "flight.search",
        (
            "find flights from {city_a} to {city_b}",
            "show nonstop flights from {city_a} to {city_b} next {day}",
            "list flights from {city_a} to {city_b} that arrive before {time}",
        ),
    ),
    LabelSpec(
        "weather.lookup",
        (
            "what is the weather in {city_a}",
            "will it rain in {city_a} on {day}",
            "show the hourly forecast for {city_a} after {time}",
        ),
    ),
    LabelSpec(
        "calendar.create",
        (
            "schedule a meeting with {person} on {day}",
            "create a calendar event with {person} at {time}",
            "book a planning session with {person} after lunch on {day}",
        ),
    ),
    LabelSpec(
        "restaurant.reserve",
        (
            "reserve a table for {number} in {city_a}",
            "find a restaurant in {city_a} for {number} people",
            "book dinner for {number} near {place} after {time}",
        ),
    ),
    LabelSpec(
        "music.play",
        (
            "play music by {artist}",
            "play the album by {artist}",
            "start a quiet playlist by {artist} after the meeting",
        ),
    ),
    LabelSpec(
        "email.send",
        (
            "send an email to {person}",
            "email {person} about the report",
            "draft a follow up email to {person} if the report is approved",
        ),
    ),
    LabelSpec(
        "navigation.route",
        (
            "navigate from {city_a} to {place}",
            "get directions to {place}",
            "route me to {place} avoiding tolls after {time}",
        ),
    ),
    LabelSpec(
        "timer.set",
        (
            "set a timer for {number} minutes",
            "remind me in {number} minutes",
            "set a timer for {number} minutes after the oven preheats",
        ),
    ),
)


def get_label_specs(num_labels: int) -> List[LabelSpec]:
    if num_labels <= len(BASE_LABEL_SPECS):
        return list(BASE_LABEL_SPECS[:num_labels])

    specs = list(BASE_LABEL_SPECS)
    for idx in range(len(BASE_LABEL_SPECS), num_labels):
        specs.append(
            LabelSpec(
                f"synthetic.intent_{idx}",
                (
                    "perform synthetic task {number} for {person}",
                    "run synthetic workflow {number} in {city_a}",
                    "compose synthetic request {number} after checking {place}",
                ),
            )
        )
    return specs


class MockSemanticParsingDataset(Dataset):
    """Synthetic utterances with semantic parse labels and hierarchy-like depth."""

    def __init__(self, num_samples: int = 512, num_labels: int = 8, seed: int = 42) -> None:
        self.num_samples = num_samples
        self.label_specs = get_label_specs(num_labels)
        self.rng = random.Random(seed)
        self.examples = [self._make_example() for _ in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]

    @property
    def label_names(self) -> List[str]:
        return [spec.name for spec in self.label_specs]

    def _make_example(self) -> Dict[str, Any]:
        label = self.rng.randrange(len(self.label_specs))
        spec = self.label_specs[label]
        template = self.rng.choice(spec.templates)

        values = {
            "artist": self.rng.choice(["Nina Simone", "Miles Davis", "Radiohead", "BTS"]),
            "city_a": self.rng.choice(["Boston", "Denver", "Seattle", "Austin", "Chicago"]),
            "city_b": self.rng.choice(["Miami", "Dallas", "Portland", "Atlanta", "Phoenix"]),
            "day": self.rng.choice(["Monday", "Friday", "Sunday", "Tuesday"]),
            "number": str(self.rng.choice([2, 3, 4, 15, 30, 45])),
            "person": self.rng.choice(["Alex", "Morgan", "Priya", "Daniel"]),
            "place": self.rng.choice(["the airport", "downtown", "the office", "the station"]),
            "time": self.rng.choice(["8 am", "noon", "5 pm", "9 pm"]),
        }
        sentence = template.format(**values)

        complexity = self.rng.choice([1, 1, 2, 2, 3, 4])
        if complexity >= 2:
            sentence += self.rng.choice(
                [
                    " and include only options with high confidence",
                    " while respecting the saved user preference",
                    " after checking the latest related constraint",
                ]
            )
        if complexity >= 3:
            sentence += self.rng.choice(
                [
                    " unless the primary condition fails",
                    " and compare it with the previous request",
                    " if the destination is still available",
                ]
            )
        if complexity >= 4:
            sentence += self.rng.choice(
                [
                    " before notifying the contact that matched the earlier filter",
                    " while excluding results that conflict with the calendar entry",
                ]
            )

        return {
            "text": sentence,
            "label": label,
            "complexity": float(complexity),
        }


class SimpleWhitespaceTokenizer:
    """Small deterministic tokenizer for offline smoke tests.

    It mimics the subset of the HuggingFace tokenizer API used by ``train.py``.
    Token ids are hash buckets, so the vocabulary size is fixed before model
    construction and no mutable vocabulary synchronization is required.
    """

    pad_token_id = 0
    cls_token_id = 101
    sep_token_id = 102

    def __init__(self, vocab_size: int = 30522) -> None:
        if vocab_size <= self.sep_token_id + 1:
            raise ValueError("vocab_size must be greater than 103")
        self.vocab_size = vocab_size
        self.pattern = re.compile(r"[A-Za-z0-9]+|[^\w\s]")

    def __call__(
        self,
        texts: Sequence[str],
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 64,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = [self._encode(text, max_length=max_length, truncation=truncation) for text in texts]
        if padding == "max_length":
            target_length = max_length
        else:
            target_length = min(max(len(ids) for ids in encoded), max_length)

        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for ids in encoded:
            ids = ids[:target_length]
            pad_len = target_length - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        if return_tensors != "pt":
            raise ValueError("SimpleWhitespaceTokenizer only supports return_tensors='pt'")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def _encode(self, text: str, max_length: int, truncation: bool) -> List[int]:
        tokens = self.pattern.findall(text.lower())
        ids = [self.cls_token_id] + [self._token_to_id(token) for token in tokens] + [self.sep_token_id]
        if truncation:
            ids = ids[:max_length]
            if ids[-1] != self.sep_token_id:
                ids[-1] = self.sep_token_id
        return ids

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        bucket_count = self.vocab_size - (self.sep_token_id + 1)
        return self.sep_token_id + 1 + (int(digest, 16) % bucket_count)


def semantic_collate_fn(
    batch: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    texts = [item["text"] for item in batch]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    output = dict(encoded)
    output["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    output["complexity"] = torch.tensor([item["complexity"] for item in batch], dtype=torch.float)
    return output
