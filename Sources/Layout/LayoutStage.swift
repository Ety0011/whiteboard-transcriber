import CoreML
import Shared

/// Stage 5 — Layout Classification (Neural Engine, ~8 ms).
/// Runs YOLOv11n (5.4 MB, FP16, 640×640) via CoreML to classify changed
/// regions as .text, .diagram, .table, or .equation.
/// Model: Models/yolo11n_layout.mlpackage — load once at startup.
/// Runs on VisionActor to serialise ANE access alongside Stage 2 and Stage 6.
public actor LayoutStage: PipelineStage {
    public typealias Input = LayoutInput
    public typealias Output = [DetectedRegion]

    public init() {}

    public func process(_ input: LayoutInput) async throws -> [DetectedRegion] {
        throw PipelineError.notImplemented
    }
}
