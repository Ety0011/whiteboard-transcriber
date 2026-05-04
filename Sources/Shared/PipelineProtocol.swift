import Foundation

public protocol PipelineStage: Sendable {
    associatedtype Input: Sendable
    associatedtype Output: Sendable
    func process(_ input: Input) async throws -> Output
}

public enum PipelineError: Error, Sendable {
    case notImplemented
    case modelLoadFailed(String)
    case processingFailed(String)
    case frameDropped
}
