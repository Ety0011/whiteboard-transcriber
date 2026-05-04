import Metal
import MetalPerformanceShaders
import Shared

/// Stage 3 — Surface Reconstruction (GPU, ~3 ms).
/// Maintains a running-median background model in a 30-frame MTLBuffer ring.
/// Pixels inside the person mask are not used to update the model; their values
/// are filled from the stored median. All allocations use .storageModeShared.
/// Runs on MetalActor.
public actor BackgroundStage: PipelineStage {
    public typealias Input = BackgroundInput
    public typealias Output = PipelineFrame

    public init() {}

    public func process(_ input: BackgroundInput) async throws -> PipelineFrame {
        throw PipelineError.notImplemented
    }
}
