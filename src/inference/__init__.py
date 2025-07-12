"""
Inference package for Khmer OCR.
Provides OCR engine and evaluation metrics for model inference.
"""
from .ocr_engine import KhmerOCREngine
from .metrics import OCRMetrics, ConfidenceMetrics, evaluate_model_predictions

__all__ = [
    "KhmerOCREngine", 
    "OCRMetrics", 
    "ConfidenceMetrics", 
    "evaluate_model_predictions"
] 