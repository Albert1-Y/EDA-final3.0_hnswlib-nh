# EDA-final3.0 — HNSW con parámetro α (alpha)

Este proyecto contiene:
- **`hnswlib_nh/`** — fork de `chroma-hnswlib` (v0.7.6) con el parámetro **α** (alpha) para poda adaptativa en la búsqueda.
- **Script de prueba** — `test_hnsw.py` para verificar integración con ChromaDB o con hnswlib directo.

---

## Sobre ChromaDB y `hnswlib`

> **ChromaDB 1.5.9** usa `import hnswlib` estático (NO dinámico). El paquete pip `chroma-hnswlib` se instala como el módulo Python `hnswlib`. `hnswlib_nh` expone `alpha` via `index.alpha` (getter/setter) gracias al binding en `python_bindings/bindings.cpp`.

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

```bash
python -c "import hnswlib; idx = hnswlib.Index('l2', 300); print('alpha por defecto:', idx.alpha)"
```

---

## 3. Instalar ChromaDB (opcional)

```bash
pip install chromadb==1.5.9
```

ChromaDB usa `import hnswlib` en tiempo de ejecución. Como el fork ya está instalado como `chroma-hnswlib` → `hnswlib`, ChromaDB lo carga automáticamente. Para verificar:

```bash
python -c "
import chromadb, hnswlib
print('ChromaDB:', chromadb.__version__)
print('hnswlib tiene alpha:', hasattr(hnswlib.Index('l2', 3), 'alpha'))
"
```

---

## 4. Descargar dataset

```bash
pip install gdown
python scripts/download_dataset.py
```

Esto crea `dataset/1990-w.npy` (228 MB) desde Google Drive.

---

## 5. Preparar dataset

Genera los archivos derivados a partir de `1990-w.npy`:

```bash
python scripts/prepare_dataset.py
```

Esto crea:
- `dataset/vectors_clean.npy`  (71 097 × 300, L2)
- `dataset/queries.npy`        (1 000 × 300)
- `dataset/ground_truth.npy`   (1 000 × 10, k=10 exacto)

---

## 6. Probar

```bash
python test_hnsw.py               # una sola query
python test_hnsw.py --all         # las 1000 queries
python test_hnsw.py --all --recall  # + recall contra ground truth
python test_hnsw.py --alpha 0.5   # con alpha personalizado
```

---

## 7. Parámetro α (alpha)

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
├── hnswlib_nh/              # fork de chroma-hnswlib
│   ├── hnswlib/             # headers C++
│   ├── src/                 # Rust FFI + bindings.cpp
│   ├── python_bindings/     # pybind11 bindings (expone alpha)
│   ├── setup.py
│   └── pyproject.toml
├── scripts/
│   ├── download_dataset.py  # descarga 1990-w.npy desde Drive
│   └── prepare_dataset.py   # genera vectors_clean, queries, ground_truth
├── test_hnsw.py
├── .gitignore
└── README.md
```
