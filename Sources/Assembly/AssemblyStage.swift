import Combine
import Foundation
import Shared

/// Stage 7 — Document Assembly (CPU, ~2 ms).
/// Maps ContentBlocks onto a spatial grid (3 cols × N rows inferred from
/// vertical spacing), deduplicates by Levenshtein ratio > 85% within 50 px,
/// emits a Markdown document, and writes it atomically via
/// FileManager.replaceItemAt. Posts a DistributedNotificationCenter event so
/// any listening UI can refresh. A Combine PassthroughSubject is also available
/// for in-process SwiftUI observers.
/// Runs on AssemblyActor to serialise document state and file writes.
public actor AssemblyStage: PipelineStage {
    public typealias Input = [ContentBlock]
    public typealias Output = URL            // path to the updated .md file

    /// Emits the output URL after every successful document update.
    public let documentUpdates = PassthroughSubject<URL, Never>()

    private let outputURL: URL

    public init(outputURL: URL) {
        self.outputURL = outputURL
    }

    public func process(_ input: [ContentBlock]) async throws -> URL {
        throw PipelineError.notImplemented
    }
}
