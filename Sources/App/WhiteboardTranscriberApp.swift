import Assembly
import Pipeline
import SwiftUI

@main
struct WhiteboardTranscriberApp: App {

    private let outputURL = URL.documentsDirectory.appending(path: "whiteboard.md")

    @State private var orchestrator: PipelineOrchestrator?

    var body: some Scene {
        WindowGroup {
            ContentView(outputURL: outputURL)
        }
        .commands {
            // TODO: add menubar commands (Start Capture, Stop Capture, Open Output)
        }
    }
}

struct ContentView: View {
    let outputURL: URL

    var body: some View {
        // TODO: embed Markdown preview pane observing AssemblyStage.documentUpdates
        Text("Whiteboard Transcriber")
            .frame(minWidth: 400, minHeight: 300)
    }
}
