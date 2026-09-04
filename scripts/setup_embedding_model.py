#!/usr/bin/env python3
"""
scripts/setup_embedding_model.py
=================================
Regenerates model_artifacts/embedding_model_ref.joblib.

That file is NOT committed to this repo (see .gitignore) -- it's an ~87MB
pickled copy of the public sentence-transformers/all-MiniLM-L6-v2 model.
Committing an 87MB binary that's trivially re-downloadable from a stable
public source (Hugging Face) just bloats every clone; regenerating it here
is faster and keeps the repo small. Every other file under model_artifacts/
(the trained SIF/rule classifiers) IS committed -- those are NOT
reproducible without the original training data and are small (<30KB each).

Run this once after `pip install -r backend/requirements.txt`, before
starting the backend:

    python scripts/setup_embedding_model.py

Safe to re-run -- it just overwrites the file.
"""
import os

import joblib
from sentence_transformers import SentenceTransformer

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = os.path.join(ARTIFACTS_DIR, "embedding_model_ref.joblib")

    print(f"Downloading {EMBEDDING_MODEL} from Hugging Face ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Writing {out_path} ...")
    joblib.dump(model, out_path)
    print("Done. backend/model.py will load this file at startup.")


if __name__ == "__main__":
    main()
