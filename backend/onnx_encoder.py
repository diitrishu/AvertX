"""
onnx_encoder.py

Lightweight sentence encoder using onnxruntime + HuggingFace tokenizer.
Replaces sentence-transformers (which pulls in PyTorch ~300 MB) with
onnxruntime (~50 MB), giving the same embeddings at a fraction of the RAM.

Produces identical 384-dim embeddings to sentence-transformers/all-MiniLM-L6-v2
via mean-pooling over token outputs (the same pooling strategy the original
SentenceTransformer uses for this model).
"""

import os
import numpy as np
from pathlib import Path

import onnxruntime as ort
from transformers import AutoTokenizer


class OnnxSentenceEncoder:
    """Drop-in replacement for SentenceTransformer.encode()."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        print(f"[onnx_encoder] Loaded from {model_dir}")

    def encode(
        self,
        sentences: list[str],
        convert_to_numpy: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode sentences into 384-dim embeddings.
        Interface matches SentenceTransformer.encode() for drop-in use.
        """
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="np",
            )

            inputs = {
                "input_ids":      encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            # token_type_ids is optional depending on model export
            if "token_type_ids" in encoded:
                inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

            outputs = self._session.run(None, inputs)
            # outputs[0] shape: (batch, seq_len, hidden) — mean pool over tokens
            token_embeddings  = outputs[0]                          # (B, T, H)
            attention_mask    = encoded["attention_mask"]           # (B, T)
            mask_expanded     = attention_mask[:, :, np.newaxis]    # (B, T, 1)
            sum_embeddings    = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask          = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_embeddings   = sum_embeddings / sum_mask           # (B, H)

            # L2-normalise (all-MiniLM-L6-v2 uses normalised embeddings)
            norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            mean_embeddings = mean_embeddings / norms

            all_embeddings.append(mean_embeddings.astype(np.float32))

        result = np.vstack(all_embeddings)
        return result
