import Shared
import Vision

/// Stage 2 — Person Segmentation (Neural Engine, ~12 ms).
/// Uses VNGeneratePersonSegmentationRequest (.balanced quality) via
/// VNSequenceRequestHandler for temporal mask smoothing across frames.
/// Output: single-channel 8-bit mask (0 = background, 255 = person/foreground).
/// Runs on VisionActor to serialise ANE access.
public actor SegmentationStage: PipelineStage {
    public typealias Input = PipelineFrame
    public typealias Output = PixelMask

    public init() {}

    public func process(_ input: PipelineFrame) async throws -> PixelMask {
        throw PipelineError.notImplemented
    }
}
