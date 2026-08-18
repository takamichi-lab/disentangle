"""Exact IIDR and retrieval metrics for DISSE embeddings.

IIDR is computed in O(ND) memory and time by accumulating normalized vectors
within source, spatial, and joint groups. This is algebraically equivalent to
materializing the full pairwise cosine-distance matrix, but avoids an N x N
allocation for the 9,216-item evaluation grid.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .cache import EMBEDDING_KEYS, validate_embedding_cache


def _normalize(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norm <= 0):
        bad = np.flatnonzero(norm.reshape(-1) <= 0)
        raise ValueError(f"Zero-norm embedding rows: {bad[:10].tolist()}")
    return array / norm


def _factorize(labels: np.ndarray, name: str) -> tuple[np.ndarray, int]:
    values = np.asarray(labels).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} is empty")
    _, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.int64, copy=False), int(inverse.max()) + 1


def _group_pair_totals(
    x: np.ndarray, inverse: np.ndarray, n_groups: int
) -> tuple[float, int]:
    """Return summed cosine distance and unordered-pair count by group."""
    counts = np.bincount(inverse, minlength=n_groups).astype(np.int64)
    sums = np.zeros((n_groups, x.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, x)
    squared_norms = np.einsum("ij,ij->i", sums, sums)

    # For unit vectors x_i, the unordered-pair cosine-distance sum is
    # (n^2 - ||sum_i x_i||^2) / 2.
    distance_sums = 0.5 * (counts.astype(np.float64) ** 2 - squared_norms)
    distance_sums = np.maximum(distance_sums, 0.0)
    pair_counts = counts * (counts - 1) // 2
    return float(distance_sums.sum()), int(pair_counts.sum())


def _conditional_distance_means(
    embeddings: np.ndarray,
    source_ids: np.ndarray,
    spatial_ids: np.ndarray,
) -> dict[str, float | int]:
    x = _normalize(embeddings)
    src, n_src = _factorize(source_ids, "source_ids")
    spa, n_spa = _factorize(spatial_ids, "spatial_ids")
    if src.shape[0] != x.shape[0] or spa.shape[0] != x.shape[0]:
        raise ValueError("Embedding and label lengths differ")

    joint_values = np.stack((src, spa), axis=1)
    _, joint = np.unique(joint_values, axis=0, return_inverse=True)
    n_joint = int(joint.max()) + 1

    src_sum, src_count = _group_pair_totals(x, src, n_src)
    spa_sum, spa_count = _group_pair_totals(x, spa, n_spa)
    joint_sum, joint_count = _group_pair_totals(x, joint, n_joint)

    same_source_diff_spatial_sum = max(0.0, src_sum - joint_sum)
    same_spatial_diff_source_sum = max(0.0, spa_sum - joint_sum)
    same_source_diff_spatial_count = src_count - joint_count
    same_spatial_diff_source_count = spa_count - joint_count

    if same_source_diff_spatial_count <= 0:
        raise ValueError("No same-source/different-spatial pairs were found")
    if same_spatial_diff_source_count <= 0:
        raise ValueError("No same-spatial/different-source pairs were found")

    d_source_fixed = (
        same_source_diff_spatial_sum / same_source_diff_spatial_count
    )
    d_spatial_fixed = (
        same_spatial_diff_source_sum / same_spatial_diff_source_count
    )
    return {
        "mean_same_source_diff_spatial": float(d_source_fixed),
        "mean_same_spatial_diff_source": float(d_spatial_fixed),
        "pairs_same_source_diff_spatial": int(same_source_diff_spatial_count),
        "pairs_same_spatial_diff_source": int(same_spatial_diff_source_count),
    }


def compute_iidr(
    embeddings: np.ndarray,
    source_ids: np.ndarray,
    spatial_ids: np.ndarray,
) -> dict[str, float | int]:
    """Compute both source and spatial IIDR values for one embedding space.

    ``IIDR_source`` is the mean distance for same-spatial/different-source
    pairs divided by the mean distance for same-source/different-spatial
    pairs. ``IIDR_spatial`` is the reciprocal ratio.
    """
    result = _conditional_distance_means(embeddings, source_ids, spatial_ids)
    d_source_fixed = float(result["mean_same_source_diff_spatial"])
    d_spatial_fixed = float(result["mean_same_spatial_diff_source"])

    if d_source_fixed == 0.0 and d_spatial_fixed == 0.0:
        iid_source = float("nan")
        iid_spatial = float("nan")
    elif d_source_fixed == 0.0:
        iid_source = float("inf")
        iid_spatial = 0.0
    elif d_spatial_fixed == 0.0:
        iid_source = 0.0
        iid_spatial = float("inf")
    else:
        iid_source = d_spatial_fixed / d_source_fixed
        iid_spatial = d_source_fixed / d_spatial_fixed

    return {
        "IIDR_source": float(iid_source),
        "IIDR_spatial": float(iid_spatial),
        **result,
    }


def iidr_report(cache: Mapping[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    """Compute Table-I-style IIDR values for every available embedding."""
    data = validate_embedding_cache(cache)
    source_ids = data["source_id"]
    spatial_ids = data["spatial_id"]
    return {
        key: compute_iidr(data[key], source_ids, spatial_ids)
        for key in EMBEDDING_KEYS
        if key in data
    }


def retrieval_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    *,
    ks: Iterable[int] = (1, 5, 10),
    exclude_diagonal: bool = False,
    chunk_size: int = 256,
) -> dict[str, float | int]:
    """Compute multi-positive Recall@K, median rank, and mean rank in chunks."""
    query = _normalize(query_embeddings)
    gallery = _normalize(gallery_embeddings)
    q_ids = np.asarray(query_ids).reshape(-1)
    g_ids = np.asarray(gallery_ids).reshape(-1)
    if query.shape[0] != q_ids.size or gallery.shape[0] != g_ids.size:
        raise ValueError("Embedding and retrieval-label lengths differ")
    if exclude_diagonal and query.shape[0] != gallery.shape[0]:
        raise ValueError("Diagonal masking requires equally sized aligned sets")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    requested_ks = tuple(sorted({int(k) for k in ks}))
    if not requested_ks or requested_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")

    ranks: list[np.ndarray] = []
    valid_rows = 0
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        similarity = query[start:stop] @ gallery.T
        positives = q_ids[start:stop, None] == g_ids[None, :]

        if exclude_diagonal:
            local = np.arange(stop - start)
            global_indices = np.arange(start, stop)
            similarity[local, global_indices] = -np.inf
            positives[local, global_indices] = False

        has_positive = positives.any(axis=1)
        if not has_positive.any():
            continue
        valid_rows += int(has_positive.sum())
        best_positive = np.max(
            np.where(positives[has_positive], similarity[has_positive], -np.inf),
            axis=1,
        )
        # Continuous similarities make exact ties rare. Counting strictly
        # greater scores gives the optimistic first-positive rank for ties.
        rank = 1 + np.sum(
            similarity[has_positive] > best_positive[:, None], axis=1
        )
        ranks.append(rank.astype(np.int64, copy=False))

    if valid_rows == 0:
        raise ValueError("No positive retrieval pairs were found")
    all_ranks = np.concatenate(ranks)
    result: dict[str, float | int] = {
        f"R@{k}": float(np.mean(all_ranks <= k)) for k in requested_ks
    }
    # torch.median, used by the paper's research code, selects the lower of
    # the two middle ranks for an even number of queries. Match that behavior
    # instead of NumPy's default arithmetic mean of the middle pair.
    middle = (all_ranks.size - 1) // 2
    result["MedR"] = float(np.partition(all_ranks, middle)[middle])
    result["MnR"] = float(np.mean(all_ranks))
    result["queries"] = int(valid_rows)
    return result


def _joint_ids(source_ids: np.ndarray, spatial_ids: np.ndarray) -> np.ndarray:
    source, _ = _factorize(source_ids, "source_ids")
    spatial, _ = _factorize(spatial_ids, "spatial_ids")
    pairs = np.stack((source, spatial), axis=1)
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return inverse


def _bidirectional(
    text: np.ndarray,
    audio: np.ndarray,
    labels: np.ndarray,
    *,
    ks: Iterable[int],
    chunk_size: int,
) -> dict[str, dict[str, float | int]]:
    return {
        "text_to_audio": retrieval_metrics(
            text, audio, labels, labels, ks=ks, chunk_size=chunk_size
        ),
        "audio_to_text": retrieval_metrics(
            audio, text, labels, labels, ks=ks, chunk_size=chunk_size
        ),
    }


def evaluate_embedding_cache(
    cache: Mapping[str, np.ndarray],
    *,
    ks: Iterable[int] = (1, 5, 10),
    chunk_size: int = 256,
    iidr_only: bool = False,
) -> dict[str, object]:
    """Compute paper-aligned IIDR and optional retrieval metrics."""
    data = validate_embedding_cache(cache)
    output: dict[str, object] = {"iidr": iidr_report(data)}
    if iidr_only:
        return output

    src = data["source_id"]
    spa = data["spatial_id"]
    available = set(data).intersection(EMBEDDING_KEYS)

    cross_modal: dict[str, object] = {}
    if {"audio_source", "text_source"} <= available:
        cross_modal["on_task_source"] = _bidirectional(
            data["text_source"], data["audio_source"], src,
            ks=ks, chunk_size=chunk_size,
        )
        cross_modal["off_task_spatial"] = _bidirectional(
            data["text_source"], data["audio_source"], spa,
            ks=ks, chunk_size=chunk_size,
        )
    if {"audio_spatial", "text_spatial"} <= available:
        cross_modal["on_task_spatial"] = _bidirectional(
            data["text_spatial"], data["audio_spatial"], spa,
            ks=ks, chunk_size=chunk_size,
        )
        cross_modal["off_task_source"] = _bidirectional(
            data["text_spatial"], data["audio_spatial"], src,
            ks=ks, chunk_size=chunk_size,
        )
    if set(EMBEDDING_KEYS) <= available:
        both = _joint_ids(src, spa)
        audio_both = np.concatenate(
            (data["audio_source"], data["audio_spatial"]), axis=1
        )
        text_both = np.concatenate(
            (data["text_source"], data["text_spatial"]), axis=1
        )
        cross_modal["both"] = _bidirectional(
            text_both, audio_both, both, ks=ks, chunk_size=chunk_size
        )
    if cross_modal:
        output["cross_modal"] = cross_modal

    intra: dict[str, object] = {}
    for modality in ("audio", "text"):
        source_key = f"{modality}_source"
        spatial_key = f"{modality}_spatial"
        modality_metrics: dict[str, object] = {}
        if source_key in available:
            source_emb = data[source_key]
            modality_metrics["on_task_source"] = retrieval_metrics(
                source_emb, source_emb, src, src, ks=ks,
                exclude_diagonal=True, chunk_size=chunk_size,
            )
            modality_metrics["off_task_spatial"] = retrieval_metrics(
                source_emb, source_emb, spa, spa, ks=ks,
                exclude_diagonal=True, chunk_size=chunk_size,
            )
        if spatial_key in available:
            spatial_emb = data[spatial_key]
            modality_metrics["on_task_spatial"] = retrieval_metrics(
                spatial_emb, spatial_emb, spa, spa, ks=ks,
                exclude_diagonal=True, chunk_size=chunk_size,
            )
            modality_metrics["off_task_source"] = retrieval_metrics(
                spatial_emb, spatial_emb, src, src, ks=ks,
                exclude_diagonal=True, chunk_size=chunk_size,
            )
        if modality_metrics:
            intra[modality] = modality_metrics
    if intra:
        output["intra_modal"] = intra
    return output
