# EDA-final3.0 — HNSW con parámetro α (alpha)

Este proyecto contiene:
- **`hnswlib_nh/`** — fork de `chroma-hnswlib` (v0.7.6) con el parámetro **α** (alpha) para poda adaptativa en la búsqueda.
- **`dataset/`** — vectors_clean.npy (71,097 × 300), queries.npy (1,000 × 300), ground_truth.npy, etc.
- **Script de prueba** — `test_hnsw.py` para verificar integración con ChromaDB o con hnswlib directo.

---

## Sobre ChromaDB y `hnswlib`

> **ChromaDB 1.5.9** usa `import hnswlib` estático (NO dinámico). El paquete pip `chroma-hnswlib` se instala como el módulo Python `hnswlib`. Tu `hnswlib_nh` expone `alpha` via `index.alpha` (getter/setter) gracias al binding en `python_bindings/bindings.cpp`.

---

## Requisitos

- Python 3.10
- C++ compiler (MSVC en Windows / GCC en Linux) — solo si recompilas

El `.pyd`/`.so` no se incluye en el repo. Cada máquina debe compilar desde fuente.

---

## 1. Crear el entorno virtual

### Windows (powershell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy
```

### Mac / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy
```

---

## 2. Compilar e instalar `hnswlib_nh`

```bash
cd hnswlib_nh
pip install pybind11 numpy
pip install .
cd ..
```

> **Nota importante (Windows):** setuptools cachea `bindings.obj`. Si solo editas `hnswalg.h`, los cambios no se reflejan en el `.pyd`. Siempre borra `build/` antes de recompilar:
> ```powershell
> if (Test-Path build) { Remove-Item -Recurse -Force build }
> ```
>
> **Nota (Mac):** El build desactiva OpenMP automáticamente (no disponible en clang por defecto). La compilación funciona igual sin paralelismo.

**Verificar instalación:**

```powershell
python -c "import hnswlib; idx = hnswlib.Index('l2', 300); print('alpha por defecto:', idx.alpha)"
```

---

## 3. Probar con el dataset

Primero descarga el dataset (ver paso 4), luego:

```bash
python test_hnsw.py               # una sola query
python test_hnsw.py --all         # las 1000 queries
python test_hnsw.py --all --recall  # + recall contra ground truth
python test_hnsw.py --alpha 0.5   # con alpha personalizado
```

---

## 4. Descargar dataset

```bash
pip install gdown
python scripts/download_dataset.py
```

Esto descarga `dataset/1990-w.npy` (228 MB) desde Google Drive. Luego genera los archivos derivados:

```bash
python scripts/prepare_dataset.py
```

---

## 5. Parámetro α (alpha)

Controla la poda adaptativa durante la búsqueda:

- **α = 0.0** — greedy clásico (solo distancia, igual al HNSW original).
- **α > 0.0** — poda si `dist(c, q) > α × min_dist_seen`. Acelera la búsqueda a costa de posible pérdida de recall.

```python
idx.alpha = 0.5   # setter
print(idx.alpha)  # getter
```

Ver `ALGO_PARAMS.md` en `hnswlib_nh/` para más detalles.

---

## Estructura del proyecto

```
EDA-final3.0/
├── dataset/
│   ├── vectors_clean.npy    # 71,097 × 300 float32
│   ├── queries.npy          # 1,000 × 300 float32
│   ├── ground_truth.npy     # ground truth
│   ├── 1990-vocab.pkl
│   └── 1990-w.npy
├── hnswlib_nh/              # fork de chroma-hnswlib
│   ├── hnswlib/             # headers C++
│   ├── src/                 # Rust FFI + bindings.cpp
│   ├── python_bindings/     # pybind11 bindings (expone alpha)
│   ├── setup.py
│   ├── pyproject.toml
│   └── hnswlib.cp310-win_amd64.pyd   # compilado pre-list
├── test_hnsw.py
├── .gitignore
└── README.md
```
