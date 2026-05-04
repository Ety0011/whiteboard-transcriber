"""
Export TrOCR-small-handwritten to CoreML for Stage 6b (Handwriting OCR fallback).

Usage:
    python Scripts/convert_trocr.py

Output:
    Models/trocr_small.mlpackage  (~130 MB, FP16)

Prerequisites:
    pip install transformers coremltools torch

Target inference latency: ~50–70 ms per text-line crop on M4 Neural Engine.
Use FP16, NOT INT8 — INT8 quantisation introduces >1% CER regression vs. the
PyTorch baseline (validated on the IAM test set).
"""

raise NotImplementedError(
    "TODO: implement encoder/decoder tracing and coremltools conversion below."
)

# -- Reference implementation (complete before removing the raise above) --
#
# import coremltools as ct
# import torch
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
#
# MODEL_ID = "microsoft/trocr-small-handwritten"
# OUTPUT_PATH = "Models/trocr_small.mlpackage"
#
# model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).eval()
# processor = TrOCRProcessor.from_pretrained(MODEL_ID)
#
# # Trace encoder (ViT / BEiT-small) and decoder (GPT-2) separately.
# # Export as MLProgram with compute_units=ALL to target the Neural Engine.
# #
# # encoder_input = torch.zeros(1, 3, 384, 384)
# # traced_encoder = torch.jit.trace(model.encoder, encoder_input)
# # mlmodel_encoder = ct.convert(
# #     traced_encoder,
# #     inputs=[ct.TensorType(shape=encoder_input.shape)],
# #     compute_units=ct.ComputeUnit.ALL,
# #     compute_precision=ct.precision.FLOAT16,
# # )
# # mlmodel_encoder.save(OUTPUT_PATH)
#
# # Validate against IAM test set: accept <= 1% CER increase vs. PyTorch baseline.
