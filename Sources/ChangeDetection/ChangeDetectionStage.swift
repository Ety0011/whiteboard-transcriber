import Accelerate
import Shared

/// Stage 4 — Change Detection (CPU/SIMD, ~2 ms). Gate stage.
/// Diffs the current composite against the previously-processed composite using
/// Accelerate vImage. Extracts bounding boxes of changed regions and computes
/// a perceptual hash per region for deduplication.
/// If hasChanges == false on the result, Stages 5–7 are skipped entirely.
/// Runs on an unstructured actor (any CPU core via Accelerate SIMD).
public actor ChangeDetectionStage: PipelineStage {
    public typealias Input = PipelineFrame
    public typealias Output = ChangeDetectionResult

    private var previousFrame: PipelineFrame?

    public init() {}

    public func process(_ input: PipelineFrame) async throws -> ChangeDetectionResult {
        throw PipelineError.notImplemented
    }
}
