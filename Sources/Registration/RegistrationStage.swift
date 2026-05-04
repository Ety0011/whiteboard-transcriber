import Metal
import Shared
import Vision

/// Stage 1 — Spatial Registration (GPU, ~5 ms).
/// Corrects camera perspective via a 3×3 homography applied as a Metal compute
/// kernel. Re-detects board corners every ~10 s; between detections, tracks
/// inter-frame motion with VNTrackOpticalFlowRequest.
/// Runs on MetalActor (MTLCommandQueue owner).
public actor RegistrationStage: PipelineStage {
    public typealias Input = PipelineFrame
    public typealias Output = PipelineFrame

    public init() {}

    public func process(_ input: PipelineFrame) async throws -> PipelineFrame {
        throw PipelineError.notImplemented
    }
}
