import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation

// MARK: - Frame wrappers
// CVPixelBuffer is a CFTypeRef and not automatically Sendable; unified memory
// makes sharing safe across actor hops without copies.

public struct PipelineFrame: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    public let timestamp: CMTime
    public let frameNumber: UInt64

    public init(pixelBuffer: CVPixelBuffer, timestamp: CMTime, frameNumber: UInt64) {
        self.pixelBuffer = pixelBuffer
        self.timestamp = timestamp
        self.frameNumber = frameNumber
    }
}

/// Single-channel 8-bit person-segmentation mask produced by Stage 2.
public struct PixelMask: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer

    public init(pixelBuffer: CVPixelBuffer) {
        self.pixelBuffer = pixelBuffer
    }
}

/// Cropped pixel buffer for a single detected region, used by Stage 6 sub-stages.
public struct RegionCrop: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    public let sourceRegion: DetectedRegion

    public init(pixelBuffer: CVPixelBuffer, sourceRegion: DetectedRegion) {
        self.pixelBuffer = pixelBuffer
        self.sourceRegion = sourceRegion
    }
}

// MARK: - Region classification

public enum RegionClass: String, Sendable, CaseIterable {
    case text
    case diagram
    case table
    case equation
}

// MARK: - Stage 5 output

public struct DetectedRegion: Sendable {
    public let boundingBox: CGRect
    public let classification: RegionClass
    public let confidence: Float

    public init(boundingBox: CGRect, classification: RegionClass, confidence: Float) {
        self.boundingBox = boundingBox
        self.classification = classification
        self.confidence = confidence
    }
}

// MARK: - Stage 4 output

public typealias RegionHash = UInt32

public struct ChangeDetectionResult: Sendable {
    public let regions: [CGRect]
    public let hashes: [RegionHash]

    public var hasChanges: Bool { !regions.isEmpty }

    public init(regions: [CGRect], hashes: [RegionHash]) {
        self.regions = regions
        self.hashes = hashes
    }

    public static let noChanges = ChangeDetectionResult(regions: [], hashes: [])
}

// MARK: - Stage 6 output

public enum ContentBlockKind: Sendable {
    case text(String, confidence: Float)
    case diagram(String)    // Mermaid or inline SVG
    case table(String)      // Markdown table syntax
}

public struct ContentBlock: Sendable {
    public let kind: ContentBlockKind
    public let boundingBox: CGRect
    public let hash: RegionHash

    public init(kind: ContentBlockKind, boundingBox: CGRect, hash: RegionHash) {
        self.kind = kind
        self.boundingBox = boundingBox
        self.hash = hash
    }
}

// MARK: - Stage 6a output

public struct TextResult: Sendable {
    public let text: String
    public let confidence: Float

    public init(text: String, confidence: Float) {
        self.text = text
        self.confidence = confidence
    }
}

// MARK: - Compound stage inputs

/// Input to Stage 3: composite frame + the person mask from Stage 2.
public struct BackgroundInput: @unchecked Sendable {
    public let frame: PipelineFrame
    public let mask: PixelMask

    public init(frame: PipelineFrame, mask: PixelMask) {
        self.frame = frame
        self.mask = mask
    }
}

/// Input to Stage 5: composite frame + the changed regions from Stage 4.
public struct LayoutInput: @unchecked Sendable {
    public let frame: PipelineFrame
    public let changes: ChangeDetectionResult

    public init(frame: PipelineFrame, changes: ChangeDetectionResult) {
        self.frame = frame
        self.changes = changes
    }
}

/// Input to Stage 6: composite frame + typed region proposals from Stage 5.
public struct RecognitionInput: @unchecked Sendable {
    public let frame: PipelineFrame
    public let regions: [DetectedRegion]

    public init(frame: PipelineFrame, regions: [DetectedRegion]) {
        self.frame = frame
        self.regions = regions
    }
}
