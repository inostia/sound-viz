import argparse
import os
import time

import pygame

from src.viz import Visualization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some audio.")
    parser.add_argument("filename", type=str, help="The path to the audio file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "pygame"],
        help="The mode to run the visualization in (video or pygame render)",
        default="video",
    )

    parser.add_argument(
        "--size",
        type=int,
        help="The size of the visualization window",
        default=720,
    )

    parser.add_argument(
        "--video-async",
        type=str,
        help="Whether to generate the video asynchronously",
        default="yes",
    )

    args = parser.parse_args()
    filename = args.filename
    mode = args.mode
    video_async = args.video_async == "yes"
    size = args.size

    # Check if the file exists
    if not os.path.exists(filename):
        print("File not found.")
        exit()

    start_time = time.time()
    viz = Visualization(filename, size)
    if mode == "pygame":
        # Start a pygame window
        screen = pygame.display.set_mode((size, size))
        viz.render(screen)
    elif mode == "video":
        viz.create_video(video_async=video_async)
    else:
        print("Invalid mode.")
        exit()

    processing_time_seconds = time.time() - start_time
    processing_time_seconds = round(processing_time_seconds, 2)  # Round to 2 decimal places
    print(f"Processing time frames took {processing_time_seconds} seconds.")
