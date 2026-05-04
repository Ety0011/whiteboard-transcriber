import Shared
import Vision

/// Stage 6a — Printed / clear-handwriting OCR (Neural Engine, ~15 ms/region).
/// Uses VNRecognizeTextRequest with .fast recognition level via
/// VNSequenceRequestHandler for temporal smoothing. Lines with confidence
/// below 0.65 are routed to HandwritingRecognitionStage (6b) as a fallback.
/// Runs on VisionActor.
public actor TextRecognitionStage: PipelineStage {
    public typealias Input = RegionCrop
    public typealias Output = TextResult

    public init() {}

    public func process(_ input: RegionCrop) async throws -> TextResult {
        throw PipelineError.notImplemented
    }
}
