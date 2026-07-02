"""
Verifica que ChromaDB funcione con el dataset y que use el hnswlib
con el parámetro alpha.
"""
import numpy as np, os, sys, time

base = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, base)

vectors = np.load(os.path.join(base, "dataset", "vectors_clean.npy"))
queries = np.load(os.path.join(base, "dataset", "queries.npy"))
gt      = np.load(os.path.join(base, "dataset", "ground_truth.npy"))
print(f"Vectores: {vectors.shape}, Queries: {queries.shape}")

import chromadb, hnswlib

client = chromadb.Client(chromadb.config.Settings(anonymized_telemetry=False))
col = client.create_collection("test_hnsw", metadata={"hnsw:space": "l2"})

t0 = time.time()
B = 1000
for i in range(0, len(vectors), B):
    end = min(i + B, len(vectors))
    col.add(ids=[str(j) for j in range(i, end)],
            embeddings=vectors[i:end].tolist())
print(f"Inserción: {time.time()-t0:.1f}s")

NQ = 200
idx = np.random.choice(len(queries), NQ, replace=False)
t0 = time.time()
res = col.query(query_embeddings=queries[idx].tolist(), n_results=10)
tq = time.time() - t0
print(f"{NQ} queries en {tq:.3f}s ({tq/NQ*1000:.2f}ms avg)")

hits, total = 0, NQ * 10
for i, r in enumerate(res["ids"]):
    pred = set(int(x) for x in r)
    truth = set(gt[idx[i], :10].tolist())
    hits += len(pred & truth)
print(f"Recall@10: {hits/total:.4f}")

# Verificar alpha
idx_hnsw = hnswlib.Index("l2", 3)
print(f"alpha expuesto: {hasattr(idx_hnsw, 'alpha')}")

client.delete_collection("test_hnsw")
print("OK")
