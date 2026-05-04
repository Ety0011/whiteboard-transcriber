// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "WhiteboardTranscriber",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "WhiteboardTranscriber", targets: ["App"]),
    ],
    targets: [

        // MARK: - App Entry Point

        .executableTarget(
            name: "App",
            dependencies: ["Pipeline", "Assembly"],
            path: "Sources/App",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Pipeline Orchestrator

        .target(
            name: "Pipeline",
            dependencies: [
                "Capture",
                "Registration",
                "Segmentation",
                "Background",
                "ChangeDetection",
                "Layout",
                "Recognition",
                "Assembly",
                "Shared",
            ],
            path: "Sources/Pipeline",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Camera Capture

        .target(
            name: "Capture",
            dependencies: ["Shared"],
            path: "Sources/Capture",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Stage 1: Spatial Registration

        .target(
            name: "Registration",
            dependencies: ["Shared"],
            path: "Sources/Registration",
            resources: [.process("Shaders")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Stage 2: Person Segmentation

        .target(
            name: "Segmentation",
            dependencies: ["Shared"],
            path: "Sources/Segmentation",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Stage 3: Surface Reconstruction

        .target(
            name: "Background",
            dependencies: ["Shared"],
            path: "Sources/Background",
            resources: [.process("Shaders")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Stage 4: Change Detection

        .target(
            name: "ChangeDetection",
            dependencies: ["Shared"],
            path: "Sources/ChangeDetection",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Stage 5: Layout Classification

        .target(
            name: "Layout",
            dependencies: ["Shared"],
            path: "Sources/Layout",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Stage 6: Content Recognition

        .target(
            name: "Recognition",
            dependencies: ["Shared"],
            path: "Sources/Recognition",
            resources: [.process("Shaders")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Stage 7: Document Assembly

        .target(
            name: "Assembly",
            dependencies: ["Shared"],
            path: "Sources/Assembly",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Shared Utilities & Protocols

        .target(
            name: "Shared",
            path: "Sources/Shared",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),

        // MARK: - Test Targets

        .testTarget(
            name: "RegistrationTests",
            dependencies: ["Registration"],
            path: "Tests/RegistrationTests"
        ),
        .testTarget(
            name: "SegmentationTests",
            dependencies: ["Segmentation"],
            path: "Tests/SegmentationTests"
        ),
        .testTarget(
            name: "BackgroundTests",
            dependencies: ["Background"],
            path: "Tests/BackgroundTests"
        ),
        .testTarget(
            name: "ChangeDetectionTests",
            dependencies: ["ChangeDetection"],
            path: "Tests/ChangeDetectionTests"
        ),
        .testTarget(
            name: "LayoutTests",
            dependencies: ["Layout"],
            path: "Tests/LayoutTests"
        ),
        .testTarget(
            name: "RecognitionTests",
            dependencies: ["Recognition"],
            path: "Tests/RecognitionTests"
        ),
        .testTarget(
            name: "AssemblyTests",
            dependencies: ["Assembly"],
            path: "Tests/AssemblyTests"
        ),
    ]
)
