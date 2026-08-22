"""
MegaGemm Quantization — INT8 and 4-bit quantized linear layers.

- Int8Linear — INT8 W8A16 per-channel quantization (2x compression)
- QuantizedLinear — AWQ W4A16 4-bit quantization (4x compression)
- NativeW4A16Linear — standalone symmetric INT4, optionally structured 2:4
"""

from .native_w4a16 import NativeW4A16Linear
from .w8a16 import Int8Linear

# AWQ requires autoawq
try:
    from .w4a16 import QuantizedLinear, W4A16_AVAILABLE
except Exception:
    W4A16_AVAILABLE = False

__all__ = ['Int8Linear', 'NativeW4A16Linear', 'QuantizedLinear', 'W4A16_AVAILABLE']
