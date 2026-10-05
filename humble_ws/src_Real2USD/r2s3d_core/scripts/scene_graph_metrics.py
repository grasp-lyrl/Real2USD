"""Scene-graph evaluation metrics"""

from __future__ import annotations

import collections
import dataclasses
import operator
import re
import typing
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

_DEFAULT_CHUNK_SIZE = 1024
_METRIC_RELATIVE_TOLERANCE = 1e-12
_METRIC_ABSOLUTE_TOLERANCE = 1e-15
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """Raised when a benchmark-domain value violates its contract."""


def _require_canonical_identifier(value: str, field_name: str) -> None:
    """Validates that ``value`` is a non-empty alphanumeric identifier."""
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValidationError(
            f"{field_name} must be a canonical identifier using letters, digits, '.', '_', ':', '/', '+', or '-'."
        )


def _finite_rate(value: object, field_name: str) -> float:
    """Validates and normalizes a value to a finite float in [0, 1]."""
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{field_name} must be a finite rate in [0, 1].")
    try:
        normalized = float(typing.cast(float, value))
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{field_name} must be a finite rate in [0, 1].") from error
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValidationError(f"{field_name} must be a finite rate in [0, 1].")
    return normalized


def _nonnegative_count(value: object, field_name: str) -> int:
    """Validates and normalizes a value to a non-negative integer."""
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{field_name} must be a non-negative integer.")
    try:
        normalized = operator.index(typing.cast(typing.SupportsIndex, value))
    except TypeError as error:
        raise ValidationError(f"{field_name} must be a non-negative integer.") from error
    if normalized < 0:
        raise ValidationError(f"{field_name} must be a non-negative integer.")
    return normalized


def _harmonic_mean(first: float, second: float) -> float:
    """Harmonic mean of two non-negative values; returns 0 when the sum is 0."""
    denominator = first + second
    return 0.0 if denominator == 0.0 else 2.0 * first * second / denominator


def _require_close_rate(value: float, expected: float, field_name: str) -> None:
    if not np.isclose(value, expected, rtol=_METRIC_RELATIVE_TOLERANCE, atol=_METRIC_ABSOLUTE_TOLERANCE):
        raise ValidationError(
            f"{field_name} is inconsistent with the metric inputs; expected {expected}, received {value}."
        )


def _validate_points(points: npt.ArrayLike, field_name: str) -> npt.NDArray[np.float64]:
    """Validates and normalizes a point array to float64 (N, 3)."""
    try:
        array = np.asarray(points)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{field_name} must be a rectangular real numeric array.") from error
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValidationError(f"{field_name} must contain real numeric values.")
    normalized = np.asarray(array, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 3:
        raise ValidationError(f"{field_name} must have shape (N, 3); received {normalized.shape}.")
    if normalized.shape[0] == 0:
        raise ValidationError(
            f"{field_name} must contain at least one point; point-cloud metrics are undefined for empty sets."
        )
    if not np.isfinite(normalized).all():
        raise ValidationError(f"{field_name} must contain only finite values.")
    return normalized


def _validate_chunk_size(chunk_size: int) -> int:
    if isinstance(chunk_size, (bool, np.bool_)):
        raise ValidationError("chunk_size must be a positive integer.")
    try:
        normalized = operator.index(chunk_size)
    except TypeError as error:
        raise ValidationError("chunk_size must be a positive integer.") from error
    if normalized <= 0:
        raise ValidationError("chunk_size must be a positive integer.")
    return normalized


def _validate_distance_threshold(distance_threshold: float) -> float:
    if isinstance(distance_threshold, (bool, np.bool_)):
        raise ValidationError("distance_threshold must be finite and strictly positive.")
    try:
        normalized = float(distance_threshold)
    except (TypeError, ValueError) as error:
        raise ValidationError("distance_threshold must be finite and strictly positive.") from error
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValidationError("distance_threshold must be finite and strictly positive.")
    return normalized


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PointCoverageMetrics:
    """Immutable bidirectional point-coverage result.

    Attributes:
        predicted_coverage: Fraction of predicted points covered by ground truth.
        ground_truth_coverage: Fraction of ground-truth points covered by predictions.
        f_score: Harmonic mean of the two directional coverages.
    """

    predicted_coverage: float
    ground_truth_coverage: float
    f_score: float

    def __post_init__(self) -> None:
        predicted_coverage = _finite_rate(self.predicted_coverage, "predicted_coverage")
        ground_truth_coverage = _finite_rate(self.ground_truth_coverage, "ground_truth_coverage")
        f_score = _finite_rate(self.f_score, "f_score")
        _require_close_rate(f_score, _harmonic_mean(predicted_coverage, ground_truth_coverage), "f_score")
        object.__setattr__(self, "predicted_coverage", predicted_coverage)
        object.__setattr__(self, "ground_truth_coverage", ground_truth_coverage)
        object.__setattr__(self, "f_score", f_score)


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectAssignment:
    """Immutable assignment of one uniquely identified detection to an object.

    Attributes:
        detection_id: Canonical identifier for the detection.
        object_id: Canonical identifier for the assigned object.
    """

    detection_id: str
    object_id: str

    def __post_init__(self) -> None:
        _require_canonical_identifier(self.detection_id, "detection_id")
        _require_canonical_identifier(self.object_id, "object_id")


@dataclasses.dataclass(frozen=True, slots=True)
class AssignmentMetrics:
    """Immutable exact-assignment precision, recall, and F1 result.

    Attributes:
        true_positives: Number of exact (detection_id, object_id) matches.
        false_positives: Predicted pairs absent from ground truth.
        false_negatives: Ground-truth pairs absent from predictions.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1: Harmonic mean of precision and recall.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def __post_init__(self) -> None:
        true_positives = _nonnegative_count(self.true_positives, "true_positives")
        false_positives = _nonnegative_count(self.false_positives, "false_positives")
        false_negatives = _nonnegative_count(self.false_negatives, "false_negatives")
        precision = _finite_rate(self.precision, "precision")
        recall = _finite_rate(self.recall, "recall")
        f1 = _finite_rate(self.f1, "f1")

        predicted_count = true_positives + false_positives
        ground_truth_count = true_positives + false_negatives
        if predicted_count == 0 and ground_truth_count == 0:
            expected_precision = expected_recall = expected_f1 = 1.0
        elif predicted_count == 0 or ground_truth_count == 0:
            expected_precision = expected_recall = expected_f1 = 0.0
        else:
            expected_precision = true_positives / predicted_count
            expected_recall = true_positives / ground_truth_count
            expected_f1 = _harmonic_mean(expected_precision, expected_recall)
        _require_close_rate(precision, expected_precision, "precision")
        _require_close_rate(recall, expected_recall, "recall")
        _require_close_rate(f1, expected_f1, "f1")

        object.__setattr__(self, "true_positives", true_positives)
        object.__setattr__(self, "false_positives", false_positives)
        object.__setattr__(self, "false_negatives", false_negatives)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "recall", recall)
        object.__setattr__(self, "f1", f1)


@dataclasses.dataclass(frozen=True, slots=True)
class TrackObservation:
    """Immutable track assignments for one fixed detection correspondence.

    Attributes:
        expected_detection_id: Canonical identifier for the expected detection.
        predicted_detection_id: Canonical identifier for the predicted detection.
        oracle_id: Canonical oracle object identifier.
        predicted_id: Canonical predicted object identifier.
    """

    expected_detection_id: str
    predicted_detection_id: str
    oracle_id: str
    predicted_id: str

    def __post_init__(self) -> None:
        _require_canonical_identifier(self.expected_detection_id, "expected_detection_id")
        _require_canonical_identifier(self.predicted_detection_id, "predicted_detection_id")
        _require_canonical_identifier(self.oracle_id, "oracle_id")
        _require_canonical_identifier(self.predicted_id, "predicted_id")


@dataclasses.dataclass(frozen=True, slots=True)
class TrackMetrics:
    """Immutable detection completeness and label-invariant conditional partition metrics.

    Detection metrics evaluate the completeness of the fixed one-to-one correspondence.
    Association, fragmentation, merge, and pairwise metrics use only corresponded detections
    and therefore evaluate track partitions conditional on detection correspondence.

    A metric is ``None`` when it has no applicable denominator.

    Attributes:
        expected_detection_count: Number of declared expected detections.
        predicted_detection_count: Number of declared predicted detections.
        matched_detection_count: Number of fixed detection correspondences.
        missed_detection_count: Expected detections without a correspondence.
        extra_detection_count: Predicted detections without a correspondence.
        detection_precision: Corresponded fraction of predicted detections.
        detection_recall: Corresponded fraction of expected detections.
        detection_f1: Harmonic mean of detection precision and recall.
        oracle_track_count: Number of oracle tracks among correspondences.
        predicted_track_count: Number of predicted tracks among correspondences.
        true_positive_pairs: Corresponded pairs grouped by both partitions.
        false_positive_pairs: Corresponded pairs grouped only by the predicted partition.
        false_negative_pairs: Corresponded pairs grouped only by the oracle partition.
        association_consistency: Fraction of detections retained after assigning each oracle track
            to its most frequent predicted track.
        association_purity: Fraction of detections retained after assigning each predicted track
            to its most frequent oracle track.
        fragmentation: Mean number of predicted tracks overlapping an oracle track.
        merge_rate: Mean number of oracle tracks overlapping a predicted track.
        pairwise_precision: Precision over predicted same-object detection pairs.
        pairwise_recall: Recall over oracle same-object detection pairs.
        pairwise_f1: Harmonic mean of pairwise precision and recall.
    """

    expected_detection_count: int
    predicted_detection_count: int
    matched_detection_count: int
    missed_detection_count: int
    extra_detection_count: int
    detection_precision: float | None
    detection_recall: float | None
    detection_f1: float | None
    oracle_track_count: int
    predicted_track_count: int
    true_positive_pairs: int
    false_positive_pairs: int
    false_negative_pairs: int
    association_consistency: float | None
    association_purity: float | None
    fragmentation: float | None
    merge_rate: float | None
    pairwise_precision: float | None
    pairwise_recall: float | None
    pairwise_f1: float | None


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------


def _directional_statistics(
    query_points: npt.NDArray[np.float64],
    reference_points: npt.NDArray[np.float64],
    *,
    chunk_size: int,
    distance_threshold: float | None,
) -> tuple[float, float | None]:
    """Chunked nearest-neighbor mean distance and optional coverage fraction.

    Both axes are chunked so the full pairwise distance matrix is never materialized.
    """
    distance_sum = 0.0
    covered_count = 0
    for query_start in range(0, query_points.shape[0], chunk_size):
        query_chunk = query_points[query_start : query_start + chunk_size]
        minimum_squared_distances = np.full(query_chunk.shape[0], np.inf, dtype=np.float64)
        for reference_start in range(0, reference_points.shape[0], chunk_size):
            reference_chunk = reference_points[reference_start : reference_start + chunk_size]
            with np.errstate(over="ignore", invalid="ignore"):
                differences = query_chunk[:, np.newaxis, :] - reference_chunk[np.newaxis, :, :]
                squared_distances = np.einsum("ijk,ijk->ij", differences, differences)
            np.minimum(minimum_squared_distances, np.min(squared_distances, axis=1), out=minimum_squared_distances)
        distances = np.sqrt(minimum_squared_distances)
        if not np.isfinite(distances).all():
            raise ValidationError("Point coordinates are finite, but their Euclidean distances overflow float64.")
        chunk_distance_sum = float(np.sum(distances, dtype=np.float64))
        distance_sum += chunk_distance_sum
        if not np.isfinite(chunk_distance_sum) or not np.isfinite(distance_sum):
            raise ValidationError(
                "Point coordinates are finite, but their accumulated nearest-neighbor distances overflow float64."
            )
        if distance_threshold is not None:
            covered_count += int(np.count_nonzero(distances <= distance_threshold))

    mean_distance = distance_sum / float(query_points.shape[0])
    if distance_threshold is None:
        return mean_distance, None
    return mean_distance, covered_count / float(query_points.shape[0])


def symmetric_chamfer_distance(
    predicted_points: npt.ArrayLike,
    ground_truth_points: npt.ArrayLike,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> float:
    """Symmetric mean Euclidean Chamfer distance between two point clouds.

    The result is the arithmetic mean of the predicted-to-ground-truth and ground-truth-to-predicted
    mean nearest-neighbor distances. Both point-set axes are chunked so no full pairwise distance
    matrix is materialized. Empty point sets are rejected because their directional means are undefined.

    Args:
        predicted_points: Predicted points with shape ``(N, 3)``.
        ground_truth_points: Ground-truth points with shape ``(M, 3)``.
        chunk_size: Maximum number of points on either pairwise-matrix axis.

    Returns:
        Symmetric mean of unsquared Euclidean nearest-neighbor distances.

    Raises:
        ValidationError: Points or ``chunk_size`` violate the metric contract.
    """
    predicted = _validate_points(predicted_points, "predicted_points")
    ground_truth = _validate_points(ground_truth_points, "ground_truth_points")
    normalized_chunk_size = _validate_chunk_size(chunk_size)

    predicted_mean, _ = _directional_statistics(predicted, ground_truth, chunk_size=normalized_chunk_size, distance_threshold=None)
    ground_truth_mean, _ = _directional_statistics(ground_truth, predicted, chunk_size=normalized_chunk_size, distance_threshold=None)
    return (predicted_mean + ground_truth_mean) / 2.0


def point_coverage_f_score(
    predicted_points: npt.ArrayLike,
    ground_truth_points: npt.ArrayLike,
    *,
    distance_threshold: float,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> PointCoverageMetrics:
    """Bidirectional point coverage and its harmonic-mean F-score.

    A point is covered when its nearest point in the other set is at a distance less than or equal
    to ``distance_threshold``. Empty point sets are rejected.

    Args:
        predicted_points: Predicted points with shape ``(N, 3)``.
        ground_truth_points: Ground-truth points with shape ``(M, 3)``.
        distance_threshold: Finite, strictly positive inclusive distance threshold.
        chunk_size: Maximum number of points on either pairwise-matrix axis.

    Returns:
        Immutable directional coverage and F-score values.

    Raises:
        ValidationError: Points, threshold, or ``chunk_size`` violate the metric contract.
    """
    predicted = _validate_points(predicted_points, "predicted_points")
    ground_truth = _validate_points(ground_truth_points, "ground_truth_points")
    normalized_threshold = _validate_distance_threshold(distance_threshold)
    normalized_chunk_size = _validate_chunk_size(chunk_size)

    _, predicted_coverage = _directional_statistics(
        predicted, ground_truth, chunk_size=normalized_chunk_size, distance_threshold=normalized_threshold
    )
    _, ground_truth_coverage = _directional_statistics(
        ground_truth, predicted, chunk_size=normalized_chunk_size, distance_threshold=normalized_threshold
    )
    assert predicted_coverage is not None
    assert ground_truth_coverage is not None
    f_score = _harmonic_mean(predicted_coverage, ground_truth_coverage)
    return PointCoverageMetrics(
        predicted_coverage=predicted_coverage,
        ground_truth_coverage=ground_truth_coverage,
        f_score=f_score,
    )


# ---------------------------------------------------------------------------
# Assignment metrics
# ---------------------------------------------------------------------------


def _index_assignments(
    assignments: Iterable[ObjectAssignment],
    field_name: str,
) -> dict[str, str]:
    """Indexes assignments by detection_id, rejecting duplicates."""
    indexed: dict[str, str] = {}
    for assignment in assignments:
        if not isinstance(assignment, ObjectAssignment):
            raise TypeError(f"{field_name} must contain only ObjectAssignment values.")
        if assignment.detection_id in indexed:
            raise ValidationError(
                f"{field_name} contains duplicate detection_id {assignment.detection_id!r}; "
                "each detection must have exactly one assignment."
            )
        indexed[assignment.detection_id] = assignment.object_id
    return indexed


def assignment_precision_recall_f1(
    predicted_assignments: Iterable[ObjectAssignment],
    ground_truth_assignments: Iterable[ObjectAssignment],
) -> AssignmentMetrics:
    """Exact detection-to-object assignment precision, recall, and F1.

    A true positive is an exact ``(detection_id, object_id)`` pair match. The two object-ID sets must
    use the same identity namespace. Detection IDs must be unique within each input; object IDs may
    repeat because many detections can belong to one object.

    When both inputs are empty, precision/recall/F1 are all ``1.0`` (empty prediction matches empty
    ground truth). When exactly one input is empty, all rates are ``0.0``.

    Args:
        predicted_assignments: Predicted detection-to-object assignments.
        ground_truth_assignments: Ground-truth detection-to-object assignments.

    Returns:
        Immutable counts and exact-assignment precision, recall, and F1.

    Raises:
        TypeError: An input contains a value other than ``ObjectAssignment``.
        ValidationError: An input contains a duplicate detection ID.
    """
    predicted = _index_assignments(predicted_assignments, "predicted_assignments")
    ground_truth = _index_assignments(ground_truth_assignments, "ground_truth_assignments")
    predicted_pairs = set(predicted.items())
    ground_truth_pairs = set(ground_truth.items())

    true_positives = len(predicted_pairs & ground_truth_pairs)
    false_positives = len(predicted_pairs - ground_truth_pairs)
    false_negatives = len(ground_truth_pairs - predicted_pairs)
    if not predicted_pairs and not ground_truth_pairs:
        precision = recall = f1 = 1.0
    elif not predicted_pairs or not ground_truth_pairs:
        precision = recall = f1 = 0.0
    else:
        precision = true_positives / len(predicted_pairs)
        recall = true_positives / len(ground_truth_pairs)
        f1 = _harmonic_mean(precision, recall)

    return AssignmentMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


# ---------------------------------------------------------------------------
# Tracking metrics
# ---------------------------------------------------------------------------


def _pair_count(group_size: int) -> int:
    """Number of unordered pairs from a group of ``group_size`` elements."""
    return group_size * (group_size - 1) // 2


def _validated_detection_ids(values: Iterable[str], field_name: str) -> frozenset[str]:
    """Validates and deduplicates a set of detection identifiers."""
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be an iterable of detection identifiers, not a string.")
    identifiers: set[str] = set()
    for identifier in values:
        _require_canonical_identifier(identifier, field_name)
        if identifier in identifiers:
            raise ValidationError(
                f"{field_name} contains duplicate identifier {identifier!r}; detection IDs must be unique."
            )
        identifiers.add(identifier)
    return frozenset(identifiers)


def compute_track_metrics(
    observations: Iterable[TrackObservation],
    *,
    expected_detection_ids: Iterable[str],
    predicted_detection_ids: Iterable[str],
) -> TrackMetrics:
    """Detection completeness and conditional track-partition metrics.

    ``observations`` describes a fixed one-to-one correspondence between declared expected and
    predicted detections. Detection precision/recall/F1 measure correspondence completeness.
    Association consistency and purity are detection-weighted dominant-overlap scores in the
    oracle-to-predicted and predicted-to-oracle directions. Fragmentation and merge rate are
    unweighted means over tracks represented in the correspondence. Pairwise metrics treat every
    unordered pair of corresponded detections as a same-object or different-object decision.

    Metrics without an applicable denominator are ``None``. Empty declared detection sets have no
    detection or conditional partition scores. A singleton correspondence has defined association,
    fragmentation, and merge scores but no pairwise scores.

    Args:
        observations: Track assignments for fixed one-to-one detection correspondences.
        expected_detection_ids: Complete unique IDs for ground-truth detections.
        predicted_detection_ids: Complete unique IDs for predicted detections.

    Returns:
        Immutable completeness and label-invariant conditional partition metrics.

    Raises:
        TypeError: An input item is not a ``TrackObservation``.
        ValidationError: IDs are invalid, duplicated, undeclared, or do not form a one-to-one
            correspondence.
    """
    expected_ids = _validated_detection_ids(expected_detection_ids, "expected_detection_ids")
    predicted_ids = _validated_detection_ids(predicted_detection_ids, "predicted_detection_ids")

    oracle_to_predicted: dict[str, collections.Counter[str]] = {}
    predicted_to_oracle: dict[str, collections.Counter[str]] = {}
    observed_expected_ids: set[str] = set()
    observed_predicted_ids: set[str] = set()

    for observation in observations:
        if not isinstance(observation, TrackObservation):
            raise TypeError("observations must contain only TrackObservation values.")
        if observation.expected_detection_id not in expected_ids:
            raise ValidationError(
                f"observation expected_detection_id {observation.expected_detection_id!r} is not declared in "
                "expected_detection_ids."
            )
        if observation.predicted_detection_id not in predicted_ids:
            raise ValidationError(
                f"observation predicted_detection_id {observation.predicted_detection_id!r} is not declared in "
                "predicted_detection_ids."
            )
        if observation.expected_detection_id in observed_expected_ids:
            raise ValidationError(
                f"observations contains duplicate expected_detection_id {observation.expected_detection_id!r}; "
                "the detection correspondence must be one-to-one."
            )
        if observation.predicted_detection_id in observed_predicted_ids:
            raise ValidationError(
                f"observations contains duplicate predicted_detection_id {observation.predicted_detection_id!r}; "
                "the detection correspondence must be one-to-one."
            )
        observed_expected_ids.add(observation.expected_detection_id)
        observed_predicted_ids.add(observation.predicted_detection_id)
        oracle_to_predicted.setdefault(observation.oracle_id, collections.Counter())[observation.predicted_id] += 1
        predicted_to_oracle.setdefault(observation.predicted_id, collections.Counter())[observation.oracle_id] += 1

    expected_detection_count = len(expected_ids)
    predicted_detection_count = len(predicted_ids)
    matched_detection_count = len(observed_expected_ids)
    missed_detection_count = expected_detection_count - matched_detection_count
    extra_detection_count = predicted_detection_count - matched_detection_count
    detection_precision = matched_detection_count / predicted_detection_count if predicted_detection_count else None
    detection_recall = matched_detection_count / expected_detection_count if expected_detection_count else None
    detection_denominator = expected_detection_count + predicted_detection_count
    detection_f1 = 2.0 * matched_detection_count / detection_denominator if detection_denominator else None

    oracle_track_count = len(oracle_to_predicted)
    predicted_track_count = len(predicted_to_oracle)
    overlap_count = sum(len(predicted_counts) for predicted_counts in oracle_to_predicted.values())

    association_consistency = (
        sum(max(predicted_counts.values()) for predicted_counts in oracle_to_predicted.values())
        / matched_detection_count
        if matched_detection_count
        else None
    )
    association_purity = (
        sum(max(oracle_counts.values()) for oracle_counts in predicted_to_oracle.values())
        / matched_detection_count
        if matched_detection_count
        else None
    )
    fragmentation = overlap_count / oracle_track_count if oracle_track_count else None
    merge_rate = overlap_count / predicted_track_count if predicted_track_count else None

    true_positive_pairs = sum(
        _pair_count(overlap_size)
        for predicted_counts in oracle_to_predicted.values()
        for overlap_size in predicted_counts.values()
    )
    oracle_positive_pairs = sum(
        _pair_count(sum(predicted_counts.values())) for predicted_counts in oracle_to_predicted.values()
    )
    predicted_positive_pairs = sum(
        _pair_count(sum(oracle_counts.values())) for oracle_counts in predicted_to_oracle.values()
    )
    false_positive_pairs = predicted_positive_pairs - true_positive_pairs
    false_negative_pairs = oracle_positive_pairs - true_positive_pairs

    pairwise_precision = true_positive_pairs / predicted_positive_pairs if predicted_positive_pairs else None
    pairwise_recall = true_positive_pairs / oracle_positive_pairs if oracle_positive_pairs else None
    pairwise_f1_denominator = 2 * true_positive_pairs + false_positive_pairs + false_negative_pairs
    pairwise_f1 = 2.0 * true_positive_pairs / pairwise_f1_denominator if pairwise_f1_denominator else None

    return TrackMetrics(
        expected_detection_count=expected_detection_count,
        predicted_detection_count=predicted_detection_count,
        matched_detection_count=matched_detection_count,
        missed_detection_count=missed_detection_count,
        extra_detection_count=extra_detection_count,
        detection_precision=detection_precision,
        detection_recall=detection_recall,
        detection_f1=detection_f1,
        oracle_track_count=oracle_track_count,
        predicted_track_count=predicted_track_count,
        true_positive_pairs=true_positive_pairs,
        false_positive_pairs=false_positive_pairs,
        false_negative_pairs=false_negative_pairs,
        association_consistency=association_consistency,
        association_purity=association_purity,
        fragmentation=fragmentation,
        merge_rate=merge_rate,
        pairwise_precision=pairwise_precision,
        pairwise_recall=pairwise_recall,
        pairwise_f1=pairwise_f1,
    )
