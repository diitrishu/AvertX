"""
export_encoder_onnx.py
Exports sentence-transformers/all-MiniLM-L6-v2 to ONNX so the backend
can use onnxruntime instead of PyTorch, saving ~200 MB RAM on Render.

Run once from project root:
    python scripts/export_encoder_onnx.py

Writes: model_artifacts/encoder.onnx  (~23 MB, committed to git)
        model_artifacts/tokenizer/     (config files, committed to git)
"""
import os
from pathlib import Path

ARTIFACTS = Path(__file__).parent.parent / "model_artifacts"
MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Exporting {MODEL_ID} to ONNX ...")

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# Export to ONNX
ort_model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

out_dir = ARTIFACTS / "onnx_encoder"
out_dir.mkdir(exist_ok=True)
ort_model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

print(f"Saved to {out_dir}")
files = list(out_dir.rglob("*"))
total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1_048_576
print(f"Total size: {total_mb:.1f} MB ({len(files)} files)")
print("Done.")
