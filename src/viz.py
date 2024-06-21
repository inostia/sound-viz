import cProfile
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Type

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.audio import Audio
from src.cache import VizCache
from src.graphs.base import BaseGraph
from src.utils import generate_unique_filename

DPI = 100


class Visualization:
    """Class for visualizing audio data."""

    filename: str
    size: int
    graph_class: Type[BaseGraph]
    bpm: int
    time_signature: str = "4/4"
    fps: float
    use_cache: bool
    clear_cache: str

    def __init__(
        self,
        filename: str,
        size: int,
        graph_class: Type[BaseGraph],
        bpm: float = None,
        time_signature: str = "4/4",
        fps: int = 30,
        use_cache: bool = True,
        clear_cache: str = "no",
    ):
        self.filename = filename
        self.size = size
        if not issubclass(graph_class, BaseGraph):
            raise ValueError("graph_class must be a subclass of BaseGraph")
        self.graph_class = graph_class
        self.bpm = bpm
        self.time_signature = time_signature
        self.fps = fps
        self.use_cache = use_cache
        self.clear_cache = clear_cache

    def save_frame(
        self,
        time_position: int,
        graph: np.ndarray | plt.Axes,
        cache: VizCache,
        bg: int = 0,
    ) -> str:
        """Save the graph to a file"""
        image_filename = f"{cache.img_cache_dir}{time_position}.png"

        if isinstance(graph, plt.Figure):
            graph.savefig(image_filename, dpi=DPI, facecolor="black")
            plt.close(graph)
            return image_filename
        if isinstance(graph, np.ndarray):
            if np.shape(graph)[2] == 4:
                # If the graph has an alpha channel, blend it with a black background
                background = np.zeros(
                    (graph.shape[0], graph.shape[1], 3), dtype=graph.dtype
                )
                background[:, :, :] = bg

                # Convert from 255 to 0, 1 range
                graph = graph / 255

                # Split the graph into RGB and alpha channels
                rgb, alpha = graph[..., :3], graph[..., 3:]

                # Alpha should be the same shape as RGB
                alpha = np.repeat(alpha, 3, axis=-1)

                # Blend the RGB channels of graph and background using the alpha channel as the mask
                blended_rgb = rgb * alpha + background * (1 - alpha)

                # Convert blended back to the range [0, 255]
                blended = np.clip(blended_rgb * 255, 0, 255).astype(np.uint8)

                # Convert the RGB to BGR
                graph = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_filename, graph)
            return image_filename
        if isinstance(graph, str) and graph == image_filename:
            return image_filename
        else:
            raise TypeError("Input must be a matplotlib Axes object or a numpy array.")

    def process_frame(
        self, time_position: int, async_mode: str, save: bool = False, *args, **kwargs
    ) -> str:
        """Process a single frame of the visualization."""
        start_time = time.time()
        cache = VizCache(self.filename, self.graph_class)
        if audio := cache.get_audio_cache_item():
            pass
        else:
            audio = Audio(self.filename, self.bpm, self.time_signature, self.fps)
            cache.save_audio_cache_item(audio)
        graph = self.graph_class(self.size, self.fps, self.use_cache).draw(
            time_position, async_mode, audio, cache, *args, **kwargs
        )
        if save:
            return self.save_frame(
                time_position, graph, cache
            ), time.time() - start_time
        return graph, time.time() - start_time

    def process_time_frames(
        self,
        func: callable,
        async_mode: str = "off",
        async_workers: int = 4,
        *args,
        **kwargs,
    ):
        """Iterate over each time frame and pass the spectrogram slice to the given function"""
        audio = Audio(self.filename, self.bpm, self.time_signature, self.fps)
        n = len(audio.times)
        cache = VizCache(self.filename, self.graph_class)
        if self.clear_cache == "all":
            cache.clear_cache()
        elif self.clear_cache == "graph":
            cache.clear_graph_cache()
        elif self.clear_cache == "img":
            cache.clear_img_cache()
        elif self.clear_cache == "audio":
            cache.clear_audio_cache()

        collection = [None] * n
        took = [0] * n
        if self.use_cache:
            # Assume the cache files are in order and complete
            num_img_cache_files = min(len(cache.img_cache_files), n)
            with ThreadPoolExecutor(max_workers=2 * async_workers) as thread_executor:
                thread_futures = {
                    thread_executor.submit(func, i, "on", *args, **kwargs): i
                    for i in range(num_img_cache_files)
                }
                if len(thread_futures) > 0:
                    print(f"Processing {len(thread_futures)} cache files...")
                for future in as_completed(thread_futures):
                    i = thread_futures[future]
                    collection[i], took[i] = future.result()
                    print(
                        f"Processed frame {i+1}/{n} from cache in {took[i]:.2f} seconds"
                    )
        else:
            num_img_cache_files = 0
        if async_mode == "on":
            with ProcessPoolExecutor(max_workers=async_workers) as process_exector:
                # Process the remaining frames
                process_futures = {
                    process_exector.submit(func, i, async_mode, *args, **kwargs): i
                    for i in range(num_img_cache_files, n)
                }
                if len(process_futures) > 0:
                    print(f"Processing {len(process_futures)} frames asynchronously...")
                for future in as_completed(process_futures):
                    i = process_futures[future]
                    collection[i], took[i] = future.result()
                    print(f"Processed frame {i+1}/{n} in {took[i]:.2f} seconds")
        else:
            if not self.use_cache or n - num_img_cache_files > 0:
                print(
                    f"Processing {abs(num_img_cache_files - n)} frames synchronously..."
                )
            for i in range(num_img_cache_files, n):
                collection[i], took[i] = func(i, async_mode, *args, **kwargs)
                print(f"Processed frame {i+1}/{n} in {took[i]:.2f} seconds")
        avg_took = sum(took) / len(took)
        return collection, avg_took

    def create_video(self, async_mode: str = "off", async_workers: int = 4) -> tuple:
        """Create a video from the images generated for each time frame and add the original audio."""

        # Gen_file and save an image for each time frame - eg 0.png, 1.png, 2.png, ...
        image_files, avg_took = self.process_time_frames(
            self.process_frame, async_mode, async_workers, save=True
        )

        # Get the size of and name image
        img = cv2.imread(image_files[0])
        height, width, layers = img.shape
        output_dir = f"output/{os.path.splitext(os.path.basename(self.filename))[0]}/"
        os.makedirs(output_dir, exist_ok=True)
        video_filename, output_filename = (
            f"{output_dir}video.mp4",
            f"{output_dir}output.mp4",
        )
        video_filename = generate_unique_filename(video_filename)
        output_filename = generate_unique_filename(output_filename)

        # Create a VideoWriter object
        print(f"Creating video from {len(image_files)} images...")
        video = cv2.VideoWriter(
            video_filename, cv2.VideoWriter_fourcc(*"avc1"), self.fps, (width, height)
        )

        # Write each image tage_filedeo
        for image_file in image_files:
            video.write(cv2.imread(image_file))

        # Release the VideoWriter
        video.release()

        # Optionally, remove the image files
        # for image_file in image_files:
        #     os.remove(image_file)

        # Add the original audio to the video
        command = [
            "ffmpeg",
            "-i",
            video_filename,
            "-i",
            self.filename,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            output_filename,
            "-y",
        ]
        subprocess.run(command, check=True)
        return video_filename, output_filename, avg_took

    def display_screen(self, time_position: str | None = "0"):
        """Display the visualization on the screen"""

        if time_position is None:
            time_position = 0
        processed_frame, _ = self.process_frame(int(time_position), "off")
        if isinstance(processed_frame, plt.Figure):
            plt.figure(
                processed_frame.number
            )  # make the figure with this number current
            plt.show()  # display the current figure
        elif isinstance(processed_frame, np.ndarray):
            plt.imshow(processed_frame)
            plt.show()

    def cprofile(self, time_position: str | None = "0"):
        """Profile the visualization"""
        cProfile.runctx(
            "self.process_frame(int(time_position), 'off')",
            globals(),
            locals(),
            filename="profile_results.profile",  # Save results to a file
            sort="cumulative",
        )
        return None
