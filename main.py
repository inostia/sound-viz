import argparse
import os
import subprocess
import time

from src.utils import parse_graph_class
from src.viz import Visualization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some audio.")
    parser.add_argument("filename", type=str, help="The path to the audio file")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "screen"],
        help="The mode to run the visualization in",
        default="screen",
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
        "--time-signature",
        type=str,
        help="The time signature of the audio file",
        default="4/4",
    )

    parser.add_argument(
        "--fps",
        type=int,
        help="The FPS of the output video",
        default=30,
    )

    parser.add_argument(
        "--cache",
        type=str,
        choices=["on", "off"],
        help="Whether to use the cache",
        default="off",
    )

    parser.add_argument(
        "--clear-cache",
        type=str,
        choices=["graph", "img", "all", "off"],
        help="Whether to clear the cache",
        default="off",
    )

    parser.add_argument(
        "--async-mode",
        type=str,
        choices=["on", "off"],
        help="Whether to use async mode",
        default="off",
    )

    parser.add_argument(
        "--async-workers",
        type=int,
        help="The maximum number of workers to use for async mode.",
        default=8,
    )
    
    parser.add_argument(
        "--time-position",
        type=str,
        help="The time position to render in screen mode. Either an integer or a slice (e.g. 0:10).",
        default=None,
    )

    parser.add_argument(
        "--graph",
        type=str,
        help="The graph to render",
        default="src.graphs.bubbles.Bubbles",
    )

    args = parser.parse_args()

    # Check if the file exists
    if not os.path.exists(args.filename):
        print("File not found.")
        exit()

    start_time = time.time()

    graph_class = parse_graph_class(args.graph)
    viz = Visualization(
        filename=args.filename,
        size=args.size,
        graph_class=graph_class,
        bpm=args.bpm,
        fps=args.fps,
        use_cache=args.cache == "on",
        clear_cache=args.clear_cache,
        time_signature=args.time_signature,
    )
    if args.time_position == "off":
        args.time_position = None
    if args.mode == "video":
        viz_video, viz_output, avg_took = viz.create_video(args.async_mode, args.async_workers)
        print(f"Output video saved to: {viz_output}")
        print(f"Average processing time per frame: {avg_took:.2f} seconds")
        subprocess.run(
            f"open {viz_output}", shell=True
        )
    elif args.mode == "screen":
        viz.display_screen(time_position=args.time_position)
    else:
        print("Invalid mode.")
        exit()

    processing_time_seconds = time.time() - start_time
    processing_time_minutes = round(processing_time_seconds / 60, 2)
    print(f"Processing time took {processing_time_minutes} minutes.")
