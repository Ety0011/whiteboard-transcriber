import Accelerate
import Shared

/// Stage 6c — Diagram vectorisation (CPU/Accelerate, ~15 ms/region).
/// Heuristic pipeline: Canny edge detection → progressive probabilistic Hough
/// transform → shape fitting (rectangles, circles, arrows) → topology graph →
/// Mermaid syntax or inline SVG output. No ML model required.
/// Runs on an unstructured actor (any CPU core via Accelerate SIMD).
public actor DiagramVectorizerStage: PipelineStage {
    public typealias Input = RegionCrop
    public typealias Output = String     // Mermaid diagram or inline SVG

    public init() {}

    public func process(_ input: RegionCrop) async throws -> String {
        throw PipelineError.notImplemented
    }
}
