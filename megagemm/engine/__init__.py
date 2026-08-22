"""
MegaGemm Engine — Inference engine with paged KV cache and continuous batching.

Components:
- InferenceEngine — Main inference API (generate text from prompts)
- Scheduler — Continuous batching scheduler
- BlockManager — Paged KV cache manager
- TieredBlockManager — GPU+CPU tiered KV cache manager
- sample_logits — Token sampling (top-p, temperature, rep penalty)
- LayerOffloadManager — GPU/CPU layer offloading
- XAIReport — Interpretability reports (top-K probs, confidence, Logit Lens)
- Deterministic mode — Bit-exact reproducible inference

All imports are lazy to avoid crashing when optional dependencies
(transformers, safetensors, etc.) are not installed.
"""


def __getattr__(name):
    if name == "InferenceEngine":
        from .engine import InferenceEngine
        return InferenceEngine
    elif name == "Scheduler":
        from .scheduler import Scheduler
        return Scheduler
    elif name == "Request":
        from .scheduler import Request
        return Request
    elif name == "RequestStatus":
        from .scheduler import RequestStatus
        return RequestStatus
    elif name == "BlockManager":
        from .kv_cache import BlockManager
        return BlockManager
    elif name == "TieredBlockManager":
        from .kv_cache import TieredBlockManager
        return TieredBlockManager
    elif name == "sample_logits":
        from .sampling import sample_logits
        return sample_logits
    elif name == "XAIReport":
        from .xai import XAIReport
        return XAIReport
    elif name == "TokenPrediction":
        from .xai import TokenPrediction
        return TokenPrediction
    elif name == "GenerationStep":
        from .xai import GenerationStep
        return GenerationStep
    elif name == "InferenceMonitor":
        from .monitor import InferenceMonitor
        return InferenceMonitor
    elif name == "DashboardServer":
        from .dashboard import DashboardServer
        return DashboardServer
    elif name == "MGXProphetLibrary":
        from .prophet import MGXProphetLibrary
        return MGXProphetLibrary
    elif name == "enable_deterministic_mode":
        from .deterministic import enable_deterministic_mode
        return enable_deterministic_mode
    elif name == "disable_deterministic_mode":
        from .deterministic import disable_deterministic_mode
        return disable_deterministic_mode
    elif name == "is_deterministic":
        from .deterministic import is_deterministic
        return is_deterministic
    raise AttributeError(f"module 'megagemm.engine' has no attribute '{name}'")


__all__ = [
    'InferenceEngine',
    'Scheduler', 'Request', 'RequestStatus',
    'BlockManager', 'TieredBlockManager',
    'sample_logits',
    'MGXProphetLibrary',
    'XAIReport', 'TokenPrediction', 'GenerationStep',
    'InferenceMonitor',
    'DashboardServer',
    'enable_deterministic_mode', 'disable_deterministic_mode', 'is_deterministic',
]
