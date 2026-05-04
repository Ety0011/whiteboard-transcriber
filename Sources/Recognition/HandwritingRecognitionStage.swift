import CoreML
import Shared

/// Stage 6b — Difficult-handwriting OCR fallback (Neural Engine, ~60 ms/line).
/// Uses TrOCR-small-handwritten (~130 MB FP16) converted to CoreML.
/// Model: Models/trocr_small.mlpackage — loaded lazily on first invocation.
/// Only called when Stage 6a confidence is below 0.65 (~10–20% of lines).
/// FP16 precision is required: INT8 quantisation introduces >1% CER regression.
/// Runs on VisionActor.
public actor HandwritingRecognitionStage: PipelineStage {
    public typealias Input = RegionCrop
    public typealias Output = String

    public init() {}

    public func process(_ input: RegionCrop) async throws -> String {
        throw PipelineError.notImplemented
    }
}
