"""Standalone video replay — raw playback sanity check using Capture.

Usage:
    python replay.py recording.mp4
    python replay.py recording.mp4 --fps 60   # override target FPS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import pygame

from src.capture.video import Capture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video file to replay")
    parser.add_argument("--fps", type=float, default=0.0,
                        help="target FPS (0 = use video native FPS)")
    parser.add_argument("--width", type=int, default=960,
                        help="display window width (default: 960)")
    args = parser.parse_args()

    cap = Capture(args.video).start()
    native_fps = cap.fps or 30.0
    target_fps = args.fps if args.fps > 0 else native_fps
    src_w, src_h = cap.frame_size or (1920, 1080)
    win_h = round(args.width * src_h / src_w)

    print(f"Source: {src_w}×{src_h} @ {native_fps:.2f} fps")
    print(f"Display: {args.width}×{win_h}  target {target_fps:.2f} fps")
    print("q / Esc — quit   space — pause")

    pygame.init()
    screen = pygame.display.set_mode((args.width, win_h), pygame.RESIZABLE)
    pygame.display.set_caption("replay")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    paused = False
    frame_n = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.stop()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    cap.stop()
                    pygame.quit()
                    return
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    cap.pause() if paused else cap.resume()

        if not paused:
            bgr = cap.try_read()
            if bgr is None and not cap.is_active:
                break
            if bgr is not None:
                frame_n += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                scaled = pygame.transform.smoothscale(surf, screen.get_size())
                screen.blit(scaled, (0, 0))
                fps_surf = font.render(
                    f"{clock.get_fps():.1f} fps  frame {frame_n}",
                    True, (0, 255, 0),
                )
                screen.blit(fps_surf, (10, 10))
                pygame.display.flip()

        clock.tick(target_fps)

    cap.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
