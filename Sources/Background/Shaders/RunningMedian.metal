#include <metal_stdlib>
using namespace metal;

/// Stage 3: running-median background model update kernel.
/// For each pixel: if the person-mask value is 0 (background), updates the
/// per-pixel running median in the model buffer. Otherwise, reads the stored
/// median to fill the composite output. Operates on 1080p single-channel float
/// data. Arguments TBD during implementation.
///
/// Expected arguments:
///   currentFrame  — rectified BGRA frame (texture2d<half, access::read>)
///   personMask    — 8-bit mask from Stage 2 (texture2d<uchar, access::read>)
///   modelBuffer   — per-pixel running median state (device float *)
///   outputFrame   — clean composite BGRA frame (texture2d<half, access::write>)
///   gid           — thread position in grid (uint2)
kernel void updateRunningMedian(
    // TODO: add texture and buffer arguments
    uint2 gid [[thread_position_in_grid]]
) {
    // TODO: implement running median update and composite write
}
