import Accelerate
import CoreML
import Shared
import Vision

/// Stage 6 — Content Recognition coordinator (ANE + CPU, ~40–80 ms).
/// Dispatches each DetectedRegion to the appropriate sub-stage in parallel
/// using TaskGroup, then aggregates results into ContentBlocks.
///
/// Sub-stage routing:
///   .text      → TextRecognitionStage (6a) → HandwritingRecognitionStage (6b) if confidence < 0.65
///   .diagram   → DiagramVectorizerStage (6c)
///   .table     → TableRecognizerStage (6d)
///   .equation  → TextRecognitionStage (6a) as fallback
///
/// Runs on VisionActor; sub-stages that need Metal switch actors internally.
public actor RecognitionStage: PipelineStage {
    public typealias Input = RecognitionInput
    public typealias Output = [ContentBlock]

    private let textRecognition = TextRecognitionStage()
    private let handwritingRecognition = HandwritingRecognitionStage()
    private let diagramVectorizer = DiagramVectorizerStage()
    private let tableRecognizer = TableRecognizerStage()

    public init() {}

    public func process(_ input: RecognitionInput) async throws -> [ContentBlock] {
        throw PipelineError.notImplemented
    }
}
