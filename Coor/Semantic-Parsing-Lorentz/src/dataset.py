"""Semantic parsing data and tokenization utilities."""

from __future__ import annotations

import json
import hashlib
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from .complexity import ComplexityFeatures, compute_linguistic_complexity


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

    def __init__(
        self,
        num_samples: int = 512,
        num_labels: int = 8,
        seed: int = 42,
        complexity_mode: str = "synthetic",
        return_complexity_features: bool = False,
    ) -> None:
        if complexity_mode not in {"synthetic", "heuristic", "spacy"}:
            raise ValueError("complexity_mode must be one of: synthetic, heuristic, spacy")
        self.num_samples = num_samples
        self.label_specs = get_label_specs(num_labels)
        self.rng = random.Random(seed)
        self.complexity_mode = complexity_mode
        self.return_complexity_features = return_complexity_features
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

        synthetic_complexity = self.rng.choice([1, 1, 2, 2, 3, 4])
        if synthetic_complexity >= 2:
            sentence += self.rng.choice(
                [
                    " and include only options with high confidence",
                    " while respecting the saved user preference",
                    " after checking the latest related constraint",
                ]
            )
        if synthetic_complexity >= 3:
            sentence += self.rng.choice(
                [
                    " unless the primary condition fails",
                    " and compare it with the previous request",
                    " if the destination is still available",
                ]
            )
        if synthetic_complexity >= 4:
            sentence += self.rng.choice(
                [
                    " before notifying the contact that matched the earlier filter",
                    " while excluding results that conflict with the calendar entry",
                ]
            )

        example: Dict[str, Any] = {
            "text": sentence,
            "label": label,
        }
        if self.complexity_mode == "synthetic":
            normalized = (float(synthetic_complexity) - 1.0) / 3.0
            example.update(
                {
                    "complexity": float(synthetic_complexity),
                    "complexity_raw": float(synthetic_complexity),
                    "complexity_normalized": normalized,
                }
            )
            if self.return_complexity_features:
                _add_complexity_feature_counts(example, sentence, self.complexity_mode)
        else:
            features = compute_linguistic_complexity(sentence, mode=self.complexity_mode)
            _add_complexity_to_example(example, features, self.return_complexity_features)
        return example


class JsonlSemanticParsingDataset(Dataset):
    """JSONL text classification / semantic parsing dataset.

    Each line must contain ``text`` and ``label``. Labels may be strings or
    integers. Missing complexity is computed from the text for heuristic/spaCy
    modes; synthetic mode uses a neutral value unless the line provides one.
    """

    def __init__(
        self,
        path: Union[str, Path],
        complexity_mode: str = "heuristic",
        return_complexity_features: bool = False,
        label_to_id: Optional[Mapping[str, int]] = None,
    ) -> None:
        if complexity_mode not in {"synthetic", "heuristic", "spacy"}:
            raise ValueError("complexity_mode must be one of: synthetic, heuristic, spacy")
        self.path = Path(path)
        self.complexity_mode = complexity_mode
        self.return_complexity_features = return_complexity_features
        raw_records = self._read_records(self.path)
        if not raw_records:
            raise ValueError(f"JSONL dataset is empty: {self.path}")

        label_values = [record["label"] for record in raw_records]
        has_string_labels = any(isinstance(label, str) for label in label_values)
        has_integer_labels = any(isinstance(label, int) and not isinstance(label, bool) for label in label_values)
        has_other_labels = any(
            not isinstance(label, str) and not (isinstance(label, int) and not isinstance(label, bool))
            for label in label_values
        )
        if has_other_labels:
            raise ValueError("JSONL labels must be strings or non-negative integers")
        if has_string_labels and has_integer_labels:
            raise ValueError("JSONL labels must be all strings or all integers; mixed labels are ambiguous")

        if has_string_labels:
            if label_to_id is None:
                unique_labels = sorted({str(label) for label in label_values})
                self.label_to_id = {label: index for index, label in enumerate(unique_labels)}
            else:
                self.label_to_id = dict(label_to_id)
            missing = sorted({str(label) for label in label_values if str(label) not in self.label_to_id})
            if missing:
                raise ValueError(f"JSONL file {self.path} contains labels missing from label_to_id: {missing}")
            self.id_to_label = {index: label for label, index in self.label_to_id.items()}
        else:
            integer_labels = [int(label) for label in label_values]
            max_label = max(integer_labels)
            if min(integer_labels) < 0:
                raise ValueError("integer labels must be non-negative")
            self.label_to_id = {str(index): index for index in range(max_label + 1)}
            self.id_to_label = {index: str(index) for index in range(max_label + 1)}

        self.examples = [self._make_example(record) for record in raw_records]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]

    @property
    def label_names(self) -> List[str]:
        return [self.id_to_label[index] for index in sorted(self.id_to_label)]

    def _make_example(self, record: Dict[str, Any]) -> Dict[str, Any]:
        text = str(record["text"])
        raw_label = record["label"]
        label = self.label_to_id[str(raw_label)] if isinstance(raw_label, str) else int(raw_label)
        example: Dict[str, Any] = {"text": text, "label": label}

        if "complexity" in record and record["complexity"] is not None:
            complexity = float(record["complexity"])
            normalized = ((complexity - 1.0) / 3.0)
            example.update(
                {
                    "complexity": max(1.0, min(4.0, complexity)),
                    "complexity_raw": complexity,
                    "complexity_normalized": max(0.0, min(1.0, normalized)),
                }
            )
            if self.return_complexity_features:
                _add_complexity_feature_counts(example, text, self.complexity_mode)
        elif self.complexity_mode == "synthetic":
            example.update(
                {
                    "complexity": 1.0,
                    "complexity_raw": 1.0,
                    "complexity_normalized": 0.0,
                }
            )
            if self.return_complexity_features:
                _add_complexity_feature_counts(example, text, self.complexity_mode)
        else:
            features = compute_linguistic_complexity(text, mode=self.complexity_mode)
            _add_complexity_to_example(example, features, self.return_complexity_features)
        return example

    @staticmethod
    def _read_records(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
                if "text" not in record or "label" not in record:
                    raise ValueError(f"line {line_number} of {path} must contain text and label fields")
                records.append(record)
        return records


def _add_complexity_to_example(
    example: Dict[str, Any],
    features: ComplexityFeatures,
    return_complexity_features: bool,
) -> None:
    example.update(
        {
            "complexity": float(features.complexity_1_to_4),
            "complexity_raw": float(features.raw_score),
            "complexity_normalized": float(features.normalized_score),
        }
    )
    if return_complexity_features:
        example["complexity_features"] = _feature_counts_dict(features)


def _add_complexity_feature_counts(example: Dict[str, Any], text: str, complexity_mode: str) -> None:
    feature_mode = "heuristic" if complexity_mode == "synthetic" else complexity_mode
    example["complexity_features"] = _feature_counts_dict(compute_linguistic_complexity(text, mode=feature_mode))


def _feature_counts_dict(features: ComplexityFeatures) -> Dict[str, Any]:
    feature_dict = asdict(features)
    feature_dict.pop("raw_score", None)
    feature_dict.pop("normalized_score", None)
    feature_dict.pop("complexity_1_to_4", None)
    return feature_dict


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
) -> Dict[str, Any]:
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
    if all("complexity" in item for item in batch):
        output["complexity"] = torch.tensor([item["complexity"] for item in batch], dtype=torch.float)
    if all("complexity_raw" in item for item in batch):
        output["complexity_raw"] = torch.tensor([item["complexity_raw"] for item in batch], dtype=torch.float)
    if all("complexity_normalized" in item for item in batch):
        output["complexity_normalized"] = torch.tensor(
            [item["complexity_normalized"] for item in batch],
            dtype=torch.float,
        )
    if all("complexity_features" in item for item in batch):
        output["complexity_features"] = [item["complexity_features"] for item in batch]
    return output
