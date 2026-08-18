"""DISSE inference and evaluation utilities."""

from .metrics import compute_iidr, evaluate_embedding_cache, iidr_report

__all__ = ["compute_iidr", "iidr_report", "evaluate_embedding_cache"]
__version__ = "0.1.0"
