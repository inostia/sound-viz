import argparse
import os
import subprocess
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
        "--bpm",
        type=int,
        help="The BPM of the audio file",
        default=None,
    )

    parser.add_argument(
        "--fps",
        type=int,
        help="The FPS of the output video",
        default=30,
    )

    parser.add_argument(
        "--use-cache",
        type=str,
        choices=["yes", "no"],
        help="Whether to use the cache",
        default="yes",
    )

    parser.add_argument(
        "--clear-cache",
        type=str,
        choices=["yes", "no"],
        help="Whether to clear the cache",
        default="no",
    )

    parser.add_argument(
        "--async-mode",
        type=str,
        choices=["on", "off"],
        help="Multi-processing or multi-threading mode.",
        default="thread",
    )

    parser.add_argument(
        "--graph",
        type=str,
        help="The graph to render",
        default="src.graphs.bubbles.Bubbles",
    )

    args = parser.parse_args()
    filename = args.filename
    mode = args.mode
    size = args.size
    use_cache = args.use_cache == "yes"
    async_mode = args.async_mode
    clear_cache = args.clear_cache == "yes"
    bpm = args.bpm
    fps = args.fps
    graph = args.graph

    # Check if the file exists
    if not os.path.exists(filename):
        print("File not found.")
        exit()

    start_time = time.time()
    # Load up the graph from the module string
    graph_module, graph_class = graph.rsplit(".", 1)
    graph_module = __import__(graph_module, fromlist=[graph_class])
    graph_class = getattr(graph_module, graph_class)

    viz = Visualization(
        filename=filename,
        size=size,
        use_cache=use_cache,
        clear_cache=clear_cache,
        bpm=bpm,
        fps=fps,
        graph_class=graph_class,
    )
    if mode == "video":
        viz_video, viz_output = viz.create_video(async_mode=async_mode)
        print(f"Output video saved to: {viz_output}")
        subprocess.run(
            f"open {viz_output}", shell=True
        )
    elif mode == "pygame":
        # Start a pygame window
        screen = pygame.display.set_mode((size, size))
        viz.render(screen)
    else:
        print("Invalid mode.")
        exit()

    processing_time_seconds = time.time() - start_time
    processing_time_minutes = round(processing_time_seconds / 60, 2)
    print(f"Processing time frames took {processing_time_minutes} minutes.")
