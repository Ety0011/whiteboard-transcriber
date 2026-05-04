/// Four global actors mirror the four GCD serial queues in the architecture.
/// Annotating a type or method with one of these actors constrains it to that
/// queue's executor, giving Swift 6 strict-concurrency enforcement of the
/// threading rules described in docs/architecture.md §4.

/// Owns AVCaptureSession and dispatches raw frames into Stage 1.
/// AVCaptureSession callbacks must run on a background thread (never main).
@globalActor public actor CameraActor {
    public static let shared = CameraActor()
}

/// Serialises all VNRequest / CoreML submissions to the Neural Engine.
/// At most one ANE model may run at a time; this actor enforces that invariant.
@globalActor public actor VisionActor {
    public static let shared = VisionActor()
}

/// Owns the MTLCommandQueue and all Metal compute dispatches.
/// All MTLBuffer allocations use .storageModeShared (unified memory, no copies).
@globalActor public actor MetalActor {
    public static let shared = MetalActor()
}

/// Serialises writes to the output Markdown document and the deduplication
/// hash table so they are never mutated concurrently.
@globalActor public actor AssemblyActor {
    public static let shared = AssemblyActor()
}
