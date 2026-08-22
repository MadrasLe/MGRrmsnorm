"""
MegaGemm Embeddings - Encoder-style embedding models with Sentence Transformers compatibility.

Components:
- EmbeddingEngine - Main embedding API for GPU/CPU encoding
- DenseSpec / PoolingSpec / SentenceTransformerSpec - Parsed metadata for ST-style models
- load_sentence_transformer_spec - Read common Sentence Transformers module layouts
"""


def __getattr__(name):
    if name == "EmbeddingEngine":
        from .engine import EmbeddingEngine
        return EmbeddingEngine
    elif name == "DenseSpec":
        from .formats import DenseSpec
        return DenseSpec
    elif name == "PoolingSpec":
        from .formats import PoolingSpec
        return PoolingSpec
    elif name == "SentenceTransformerSpec":
        from .formats import SentenceTransformerSpec
        return SentenceTransformerSpec
    elif name == "load_sentence_transformer_spec":
        from .formats import load_sentence_transformer_spec
        return load_sentence_transformer_spec
    raise AttributeError(f"module 'megagemm.embeddings' has no attribute '{name}'")


__all__ = [
    "EmbeddingEngine",
    "DenseSpec",
    "PoolingSpec",
    "SentenceTransformerSpec",
    "load_sentence_transformer_spec",
]
