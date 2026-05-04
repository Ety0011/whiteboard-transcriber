#include <metal_stdlib>
using namespace metal;

/// Stage 1: perspective warp kernel.
/// Applies a 3×3 homography to remap a 1080p BGRA frame to a board-aligned
/// coordinate system. Arguments TBD during implementation.
///
/// Expected arguments:
///   inputTexture  — source BGRA frame (texture2d<half, access::sample>)
///   outputTexture — rectified BGRA frame (texture2d<half, access::write>)
///   homography    — row-major 3×3 float matrix (constant float3x3 &)
///   gid           — thread position in grid (uint2)
kernel void perspectiveWarp(
    // TODO: add texture and buffer arguments
    uint2 gid [[thread_position_in_grid]]
) {
    // TODO: implement bilinear-sampled perspective remap
}
