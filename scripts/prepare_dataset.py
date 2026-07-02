"""
Genera vectors_clean.npy, queries.npy y ground_truth.npy
a partir de 1990-w.npy (descargado desde Google Drive).
"""
import numpy as np, os, time

base = os.path.join(os.path.dirname(__file__), "..", "dataset")
raw = np.load(os.path.join(base, "1990-w.npy")).astype(np.float32)

# Filtrar norma cero
norms = np.linalg.norm(raw, axis=1)
vectors = raw[norms > 0]
print(f"{len(raw)} total -> {len(vectors)} tras filtrar {(norms == 0).sum()} zero-norm")

N_QUERIES, K = 1000, 10
np.random.seed(42)
idx_q = np.random.choice(len(vectors), N_QUERIES, replace=False)
noise = np.random.randn(N_QUERIES, vectors.shape[1]).astype(np.float32) * 0.01
queries = vectors[idx_q] + noise

# Ground truth brute-force
gt = np.empty((N_QUERIES, K), dtype=np.int32)
t0 = time.time()
for i, q in enumerate(queries):
    d = np.sum((vectors - q) ** 2, axis=1)
    gt[i] = np.argpartition(d, K)[:K]
    if (i+1) % 200 == 0: print(f"  GT: {i+1}/{N_QUERIES}")
print(f"GT brute-force: {time.time()-t0:.1f}s")

np.save(os.path.join(base, "vectors_clean.npy"), vectors)
np.save(os.path.join(base, "queries.npy"), queries)
np.save(os.path.join(base, "ground_truth.npy"), gt)
print(f"Guardado en {base}")
