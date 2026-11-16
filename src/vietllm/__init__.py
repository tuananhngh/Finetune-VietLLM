"""vietllm – fine-tuning toolkit for Vietnamese large language models."""

import os

# Let PyTorch run the few ops Apple's MPS backend lacks on the CPU instead of crashing.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.2.0"
