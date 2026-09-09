"""Evaluation metrics calculation
for Recall@k and IoU character span overlap."""

from typing import Dict, List
from src.models.question import AnsweredQuestion
from src.models.results import MinimalSearchResults
from src.models.source import MinimalSource


def compute_iou(s1: MinimalSource, s2: MinimalSource) -> float:
    """Compute Intersection over Union (IoU) for two character spans.

    Args:
        s1: First source location.
        s2: Second source location.

    Returns:
        IoU value between 0.0 and 1.0 (returns 0.0 if file paths differ).
    """
    if s1.file_path != s2.file_path:
        return 0.0

    start1, end1 = s1.first_character_index, s1.last_character_index
    start2, end2 = s2.first_character_index, s2.last_character_index

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    intersection = max(0, intersection_end - intersection_start)
    if intersection == 0:
        return 0.0

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = max(0, union_end - union_start)

    return (intersection / union) if union > 0 else 0.0


def is_source_retrieved(
    ground_truth: MinimalSource,
    retrieved_sources: List[MinimalSource],
    iou_threshold: float = 0.05,
) -> bool:
    """Check if a ground-truth source is matched by any retrieved source.

    Args:
        ground_truth: The reference source snippet.
        retrieved_sources: The list of top-k retrieved snippets.
        iou_threshold: Minimum IoU required (default 0.05).

    Returns:
        True if at least one retrieved source overlaps with IoU >= threshold.
    """
    for candidate in retrieved_sources:
        if compute_iou(ground_truth, candidate) >= iou_threshold:
            return True
    return False


def calculate_recall_at_k(
    ground_truth_questions: List[AnsweredQuestion],
    student_results: List[MinimalSearchResults],
    k_values: List[int],
    iou_threshold: float = 0.05,
) -> Dict[int, float]:
    """Calculate Recall@k across all evaluation questions
    for specified k values.

    Args:
        ground_truth_questions: Ground-truth answered questions with sources.
        student_results: Model/retriever predicted search results.
        k_values: List of k cutoffs to evaluate (e.g. [1, 3, 5, 10]).
        iou_threshold: Minimum IoU for a match.

    Returns:
        Dictionary mapping k -> Recall@k score.
    """
    results_map: Dict[str, MinimalSearchResults] = {
        r.question_id: r for r in student_results
    }

    recall_scores: Dict[int, float] = {k: 0.0 for k in k_values}
    valid_questions_count = 0

    for gt_q in ground_truth_questions:
        if not gt_q.sources:
            continue

        valid_questions_count += 1
        predicted = results_map.get(gt_q.question_id)
        predicted_sources = predicted.retrieved_sources if predicted else []

        for k in k_values:
            top_k_candidates = predicted_sources[:k]
            matched_sources_count = sum(
                1
                for source in gt_q.sources
                if is_source_retrieved(source, top_k_candidates, iou_threshold)
            )
            # Recall for this question: fraction of ground-truth sources found
            q_recall = matched_sources_count / len(gt_q.sources)
            recall_scores[k] += q_recall

    if valid_questions_count == 0:
        return {k: 0.0 for k in k_values}

    return {k: (recall_scores[k] / valid_questions_count) for k in k_values}
