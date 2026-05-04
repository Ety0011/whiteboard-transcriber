import Accelerate
import Metal
import Shared
import Vision

/// Stage 6d — Table recognition (GPU + Neural Engine, ~20 ms/table).
/// Grid detection via Hough lines runs as a Metal kernel (see Shaders/GridDetection.metal).
/// Each extracted cell is submitted to VNRecognizeTextRequest in batch.
/// Output: Markdown table syntax (| col | col | …).
/// Runs on MetalActor for the Hough dispatch; Vision OCR calls switch to VisionActor.
public actor TableRecognizerStage: PipelineStage {
    public typealias Input = RegionCrop
    public typealias Output = String    // Markdown table

    public init() {}

    public func process(_ input: RegionCrop) async throws -> String {
        throw PipelineError.notImplemented
    }
}
