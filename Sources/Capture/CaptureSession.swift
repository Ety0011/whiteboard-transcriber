import AVFoundation
import Shared

/// Wraps AVCaptureSession and delivers raw CMSampleBuffers into the pipeline.
/// start() and stop() must be called from a background thread — never the main thread.
/// The AVCaptureVideoDataOutputSampleBufferDelegate implementation must call the
/// delegate on the CameraActor queue.
public final class CaptureSession: NSObject, @unchecked Sendable {

    public weak var delegate: CaptureSessionDelegate?

    private var captureSession: AVCaptureSession?

    public override init() {
        super.init()
    }

    /// Configures and starts the AVCaptureSession.
    /// Call on a background thread (AVCaptureSession requirement).
    @CameraActor
    public func start() throws {
        throw PipelineError.notImplemented
    }

    /// Stops the AVCaptureSession.
    /// Call on a background thread (AVCaptureSession requirement).
    @CameraActor
    public func stop() {
        // TODO: captureSession?.stopRunning()
    }
}

public protocol CaptureSessionDelegate: AnyObject, Sendable {
    func captureSession(_ session: CaptureSession, didOutput frame: PipelineFrame)
}
