"""
Test script for hnswlib_nh with dataset.

Usage:
    python test_hnsw.py              # single query
    python test_hnsw.py --all        # all 1000 queries
    python test_hnsw.py --all --recall  # all queries + recall against ground_truth
    python test_hnsw.py --alpha 0.5  # set custom alpha
"""

import argparse
import os
import sys
import time

import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def load_dataset():
    vectors = np.load(os.path.join(DATASET_DIR, "vectors_clean.npy"))
    queries = np.load(os.path.join(DATASET_DIR, "queries.npy"))
    ground_truth = None
    gt_path = os.path.join(DATASET_DIR, "ground_truth.npy")
    if os.path.exists(gt_path):
        ground_truth = np.load(gt_path)
    return vectors, queries, ground_truth


def build_index(vectors, alpha=0.0):
    import hnswlib

    n, dim = vectors.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    idx.set_ef(100)
    idx.alpha = alpha
    print(f"Indexing {n} vectors (dim={dim})...", end=" ", flush=True)
    t0 = time.perf_counter()
    idx.add_items(vectors)
    t1 = time.perf_counter()
    print(f"done in {t1 - t0:.2f}s")
    return idx


def run_single_query(idx, queries):
    q = queries[0:1]
    t0 = time.perf_counter()
    labels, distances = idx.knn_query(q, k=10)
    t1 = time.perf_counter()
    print(f"\nSingle query: {t1 - t0:.4f}s")
    print(f"Top-10 labels: {labels[0]}")
    print(f"Top-10 distances: {distances[0]}")


def run_all_queries(idx, queries, ground_truth=None):
    k = 10
    t0 = time.perf_counter()
    labels, distances = idx.knn_query(queries, k=k)
    t1 = time.perf_counter()
    total = queries.shape[0]
    avg = (t1 - t0) / total
    print(f"\nAll {total} queries: {t1 - t0:.2f}s total, {avg * 1000:.2f}ms avg")

    if ground_truth is not None:
        gt = ground_truth[:, :k]
        correct = 0
        total_gt = 0
        for i in range(total):
            pred_set = set(labels[i])
            gt_set = set(gt[i])
            correct += len(pred_set & gt_set)
            total_gt += k
        recall = correct / total_gt * 100
        print(f"Recall@{k}: {recall:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Test hnswlib_nh with dataset")
    parser.add_argument("--all", action="store_true", help="Run all 1000 queries")
    parser.add_argument("--recall", action="store_true", help="Compute recall against ground_truth")
    parser.add_argument("--alpha", type=float, default=0.0, help="Set alpha parameter (default: 0.0)")
    args = parser.parse_args()

    vectors, queries, ground_truth = load_dataset()
    print(f"Vectors: {vectors.shape}, Queries: {queries.shape}")
    if ground_truth is not None:
        print(f"Ground truth: {ground_truth.shape}")

    idx = build_index(vectors, alpha=args.alpha)

    if args.all:
        gt_for_recall = ground_truth if args.recall else None
        run_all_queries(idx, queries, gt_for_recall)
    else:
        run_single_query(idx, queries)


if __name__ == "__main__":
    main()
