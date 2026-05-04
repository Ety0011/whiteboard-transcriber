"""
Export YOLOv11n to CoreML for Stage 5 (Layout Classification).

Usage:
    python Scripts/convert_yolo.py

Output:
    Models/yolo11n_layout.mlpackage

Prerequisites:
    pip install ultralytics coremltools
    # Download or train yolov11n.pt fine-tuned on whiteboard layout dataset.
    # Pre-train on DocLayNet, then fine-tune on ~2000 annotated whiteboard photos.
    # Classes: text_block, diagram, table, equation

Target inference latency: ~8 ms on M4 Neural Engine (CoreML FP16).
"""

raise NotImplementedError(
    "TODO: place yolov11n.pt in the project root, then run:\n"
    "  yolo export model=yolov11n.pt format=coreml imgsz=640 half=True nms=False\n"
    "  mv yolov11n.mlpackage Models/yolo11n_layout.mlpackage"
)

# Full export command (run directly with ultralytics CLI):
#   yolo export model=yolov11n.pt format=coreml imgsz=640 half=True nms=False
#
# Validate the exported model against a held-out whiteboard test set before
# committing Models/yolo11n_layout.mlpackage. Target mAP50 >= 0.80 on the
# whiteboard layout dataset.
