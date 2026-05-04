import Foundation
import Shared
import Assembly
import Background
import Capture
import ChangeDetection
import Layout
import Recognition
import Registration
import Segmentation

/// Chains Stages 1–7 for each camera frame.
///
/// Frame pacing:
///   - Processes every 3rd frame under normal conditions (10 Hz effective rate).
///   - Drops to every 6th frame (5 Hz) if Stage 6 takes > 200 ms.
///   - If Stage 4 reports no changes, Stages 5–7 are skipped (idle power saving).
///   - Back-pressure: if a frame is already in-flight, the incoming frame is dropped.
///
/// Stages 1–3 are pipelined (frame N+1 enters Stage 1 while N is in Stage 2).
/// All inter-stage calls cross actor boundaries via await.
public actor PipelineOrchestrator {

    private let capture: CaptureSession
    private let registration: RegistrationStage
    private let segmentation: SegmentationStage
    private let background: BackgroundStage
    private let changeDetection: ChangeDetectionStage
    private let layout: LayoutStage
    private let recognition: RecognitionStage
    private let assembly: AssemblyStage

    private var isProcessing = false
    private var framePacingCounter: UInt64 = 0

    public init(outputURL: URL) {
        capture = CaptureSession()
        registration = RegistrationStage()
        segmentation = SegmentationStage()
        background = BackgroundStage()
        changeDetection = ChangeDetectionStage()
        layout = LayoutStage()
        recognition = RecognitionStage()
        assembly = AssemblyStage(outputURL: outputURL)
    }

    /// Starts the capture session and begins processing frames.
    public func start() async throws {
        throw PipelineError.notImplemented
    }

    /// Stops the capture session and drains in-flight work.
    public func stop() async {
        // TODO: stop capture, await in-flight frame
    }

    /// Full pipeline for a single frame. Called from the CaptureSession delegate.
    /// Drops the frame immediately if a previous frame is still in flight.
    private func processFrame(_ frame: PipelineFrame) async {
        guard !isProcessing else { return }
        framePacingCounter += 1
        guard framePacingCounter % 3 == 0 else { return }

        isProcessing = true
        defer { isProcessing = false }

        do {
            // Stage 1 — Spatial Registration (MetalActor)
            let rectified = try await registration.process(frame)

            // Stage 2 — Person Segmentation (VisionActor)
            let mask = try await segmentation.process(rectified)

            // Stage 3 — Surface Reconstruction (MetalActor)
            let composite = try await background.process(BackgroundInput(frame: rectified, mask: mask))

            // Stage 4 — Change Detection (CPU) — gate
            let changes = try await changeDetection.process(composite)
            guard changes.hasChanges else { return }

            // Stage 5 — Layout Classification (VisionActor)
            let regions = try await layout.process(LayoutInput(frame: composite, changes: changes))

            // Stage 6 — Recognition (VisionActor + CPU, parallel internally)
            let blocks = try await recognition.process(RecognitionInput(frame: composite, regions: regions))

            // Stage 7 — Document Assembly (AssemblyActor)
            _ = try await assembly.process(blocks)

        } catch PipelineError.notImplemented {
            // Expected during scaffolding; stages are not yet implemented.
        } catch {
            // TODO: surface errors via a Combine publisher or os_log
        }
    }
}
