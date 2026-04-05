import random
import re
import string
from typing import Callable, Iterable, Optional, Sequence


def normalize_boundary_positions(boundaries: Iterable[int]) -> set[int]:
    return {int(boundary) for boundary in boundaries}


def boundary_positions_from_segments(segments: Sequence[str]) -> set[int]:
    positions = set()
    cursor = 0

    for segment in segments[:-1]:
        cursor += len(segment)
        positions.add(cursor)

    return positions


def whitespace_segments_with_attached_space(text: str) -> list[str]:
    parts = re.findall(r"\S+\s*|\s+", text)
    segments: list[str] = []

    for part in parts:
        if part.isspace():
            if segments:
                segments[-1] += part
            else:
                segments.append(part)
        else:
            segments.append(part)

    return [segment for segment in segments if segment]


def whitespace_boundary_positions(text: str) -> set[int]:
    return boundary_positions_from_segments(whitespace_segments_with_attached_space(text))


def boundary_confusion(predicted_boundaries: Iterable[int], gold_boundaries: Iterable[int]) -> dict[str, int]:
    predicted = normalize_boundary_positions(predicted_boundaries)
    gold = normalize_boundary_positions(gold_boundaries)

    true_positive = len(predicted & gold)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def boundary_precision_recall_f1(predicted_boundaries: Iterable[int], gold_boundaries: Iterable[int]) -> dict[str, float]:
    counts = boundary_confusion(predicted_boundaries, gold_boundaries)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **counts,
    }


def corpus_boundary_f1(
    predicted_boundary_sets: Sequence[Iterable[int]],
    gold_boundary_sets: Sequence[Iterable[int]],
) -> dict[str, float]:
    if len(predicted_boundary_sets) != len(gold_boundary_sets):
        raise ValueError("predicted_boundary_sets and gold_boundary_sets must have the same length.")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for predicted, gold in zip(predicted_boundary_sets, gold_boundary_sets):
        counts = boundary_confusion(predicted, gold)
        total_tp += counts["tp"]
        total_fp += counts["fp"]
        total_fn += counts["fn"]

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def segmentation_f1_from_segments(
    predicted_segments_list: Sequence[Sequence[str]],
    gold_segments_list: Sequence[Sequence[str]],
) -> dict[str, float]:
    predicted_boundaries = [boundary_positions_from_segments(segments) for segments in predicted_segments_list]
    gold_boundaries = [boundary_positions_from_segments(segments) for segments in gold_segments_list]
    return corpus_boundary_f1(predicted_boundaries, gold_boundaries)


def _default_noise_alphabet() -> str:
    return string.ascii_lowercase


def corrupt_text(
    text: str,
    char_error_rate: float = 0.05,
    alphabet: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    if not 0.0 <= char_error_rate <= 1.0:
        raise ValueError("char_error_rate must be between 0 and 1.")

    alphabet = alphabet or _default_noise_alphabet()
    rng = random.Random(seed)
    corrupted: list[str] = []
    idx = 0

    while idx < len(text):
        ch = text[idx]

        if ch.isspace() or rng.random() >= char_error_rate:
            corrupted.append(ch)
            idx += 1
            continue

        operation = rng.choice(["substitute", "delete", "insert", "swap"])

        if operation == "substitute":
            corrupted.append(rng.choice(alphabet))
            idx += 1
        elif operation == "delete":
            idx += 1
        elif operation == "insert":
            corrupted.append(rng.choice(alphabet))
            corrupted.append(ch)
            idx += 1
        else:
            if idx + 1 < len(text) and not text[idx + 1].isspace():
                corrupted.append(text[idx + 1])
                corrupted.append(ch)
                idx += 2
            else:
                corrupted.append(ch)
                idx += 1

    return "".join(corrupted)


def make_noisy_corpus(
    texts: Sequence[str],
    char_error_rate: float = 0.05,
    alphabet: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[str]:
    rng = random.Random(seed)
    return [
        corrupt_text(
            text,
            char_error_rate=char_error_rate,
            alphabet=alphabet,
            seed=rng.randint(0, 10**9),
        )
        for text in texts
    ]


def evaluate_perplexity(
    texts: Sequence[str],
    perplexity_fn: Callable[[Sequence[str]], float],
    max_items: Optional[int] = None,
) -> float:
    subset = list(texts[:max_items]) if max_items is not None else list(texts)
    if not subset:
        raise ValueError("evaluate_perplexity requires at least one text.")
    return float(perplexity_fn(subset))


def evaluate_noisy_perplexity(
    texts: Sequence[str],
    perplexity_fn: Callable[[Sequence[str]], float],
    char_error_rate: float = 0.05,
    alphabet: Optional[str] = None,
    seed: Optional[int] = None,
    max_items: Optional[int] = None,
) -> float:
    subset = list(texts[:max_items]) if max_items is not None else list(texts)
    noisy_texts = make_noisy_corpus(
        subset,
        char_error_rate=char_error_rate,
        alphabet=alphabet,
        seed=seed,
    )
    return float(perplexity_fn(noisy_texts))


def evaluate_ood_perplexity(
    ood_texts: Sequence[str],
    perplexity_fn: Callable[[Sequence[str]], float],
    max_items: Optional[int] = None,
) -> float:
    return evaluate_perplexity(ood_texts, perplexity_fn, max_items=max_items)


def evaluate_all_metrics(
    *,
    in_domain_texts: Sequence[str],
    perplexity_fn: Callable[[Sequence[str]], float],
    predicted_boundary_sets: Optional[Sequence[Iterable[int]]] = None,
    gold_boundary_sets: Optional[Sequence[Iterable[int]]] = None,
    noisy_char_error_rate: float = 0.05,
    noise_alphabet: Optional[str] = None,
    noise_seed: Optional[int] = None,
    ood_texts: Optional[Sequence[str]] = None,
    max_items: Optional[int] = None,
) -> dict[str, object]:
    results: dict[str, object] = {
        "ppl": evaluate_perplexity(
            in_domain_texts,
            perplexity_fn,
            max_items=max_items,
        ),
        "noisy_ppl": evaluate_noisy_perplexity(
            in_domain_texts,
            perplexity_fn,
            char_error_rate=noisy_char_error_rate,
            alphabet=noise_alphabet,
            seed=noise_seed,
            max_items=max_items,
        ),
    }

    if ood_texts is not None:
        results["ood_ppl"] = evaluate_ood_perplexity(
            ood_texts,
            perplexity_fn,
            max_items=max_items,
        )

    if predicted_boundary_sets is not None and gold_boundary_sets is not None:
        results["boundary_f1"] = corpus_boundary_f1(
            predicted_boundary_sets,
            gold_boundary_sets,
        )

    return results
