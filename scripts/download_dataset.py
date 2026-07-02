"""Descarga el dataset original desde Google Drive."""
import subprocess, sys, os

url = "https://drive.google.com/drive/folders/1_Os0gDQNpJ9bAM8fN5-Z5KxSfjzc3fLO"
out = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(out, exist_ok=True)

subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
import gdown
gdown.download_folder(url, output=out)
print(f"Dataset descargado en {out}")
