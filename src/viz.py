import hashlib
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

plt.rcParams["figure.facecolor"] = "black"
plt.rcParams["axes.facecolor"] = "black"


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

    def process_frame(self, time_position: int, save: bool = False) -> str:
        """Process a single frame of the visualization."""
        # TODO: Migrate this to the graph class
        start_time = time.time()
        audio = Audio(self.filename, self.bpm, self.time_signature, self.fps)
        cache = VizCache(self.filename, len(audio.times), self.graph_class)
        graph = self.graph_class(self.size, self.fps, self.use_cache).draw(
            time_position, audio, cache
        )
        if save:
            return self.save_frame(
                time_position, graph, cache
            ), time.time() - start_time
        return graph, time.time() - start_time

    def save_frame_orig(self, time_position: int, graph: np.ndarray, cache: VizCache) -> str:
        """Save the graph to a file"""
        image_filename = f"{cache.img_cache_dir}{time_position}.png"
        cv2.imwrite(image_filename, graph)
        return image_filename

    def save_frame(self, time_position: int, graph: np.ndarray, cache: VizCache, bg: int = 0) -> str:
        """Save the graph to a file"""
        # Create an RGB black image of the same size as graph
        background = np.zeros((graph.shape[0], graph.shape[1], 3), dtype=graph.dtype)
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

        image_filename = f"{cache.img_cache_dir}{time_position}.png"
        
        # Convert the RGB to BGR
        blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_filename, blended)
        return image_filename

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
        cache = VizCache(self.filename, n, self.graph_class)
        if self.clear_cache == "all":
            cache.clear_cache()
        elif self.clear_cache == "graph":
            cache.clear_graph_cache()
        elif self.clear_cache == "img":
            cache.clear_img_cache()

        collection = [None] * n
        took = [0] * n
        if self.use_cache:
            # Assume the cache files are in order and complete
            num_graph_cache_files = min(len(cache.graph_cache_files), n)
            with ThreadPoolExecutor(max_workers=2 * async_workers) as thread_executor:
                thread_futures = {
                    thread_executor.submit(func, i, *args, **kwargs): i
                    for i in range(num_graph_cache_files)
                }
                if len(thread_futures) > 0:
                    print(f"Processing {len(thread_futures)} cache files...")
                for future in as_completed(thread_futures):
                    i = thread_futures[future]
                    collection[i], took[i] = future.result()
                    print(
                        f"Processed frame {i+1}/{n} from graph cache in {took[i]:.2f} seconds"
                    )
        else:
            num_graph_cache_files = 0
        if async_mode == "on":
            with ProcessPoolExecutor(max_workers=async_workers) as process_exector:
                # Process the remaining frames
                process_futures = {
                    process_exector.submit(func, i, *args, **kwargs): i
                    for i in range(num_graph_cache_files, n)
                }
                if len(process_futures) > 0:
                    print(f"Processing {len(process_futures)} frames asynchronously...")
                for future in as_completed(process_futures):
                    i = process_futures[future]
                    collection[i], took[i] = future.result()
                    print(f"Processed frame {i+1}/{n} in {took[i]:.2f} seconds")
        else:
            if not self.use_cache or n - num_graph_cache_files > 0:
                print(
                    f"Processing {abs(num_graph_cache_files - n)} frames synchronously..."
                )
            for i in range(num_graph_cache_files, n):
                collection[i], took[i] = func(i, *args, **kwargs)
                print(f"Processed frame {i+1}/{n} in {took[i]:.2f} seconds")
        avg_took = sum(took) / len(took)
        return collection, avg_took

    def generate_unique_filename(self, filename) -> str:
        """Generate a unique filename by appending a hash if the file already exists"""
        if os.path.exists(filename):
            output_dir = os.path.dirname(filename)
            # Use the last modified file in the directory as the seed
            hash_seed = max(
                [os.path.join(output_dir, f) for f in os.listdir(output_dir)],
                key=os.path.getmtime,
            )
            hash = hashlib.md5(hash_seed.encode())
            basename, ext = os.path.splitext(filename)
            return f"{basename}_{hash.hexdigest()[:5]}{ext}"
        return filename

    def create_video(self, async_mode: str = "off", async_workers: int = 4) -> tuple:
        """Create a video from the images generated for each time frame and add the original audio."""
        # Gen_file and save an image for each time frame
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
        video_filename = self.generate_unique_filename(video_filename)
        output_filename = self.generate_unique_filename(output_filename)

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

    def screen(self):
        """Display the visualization on the screen"""
        # i = 42
        i = 0
        graph, _ = self.process_frame(i)
        # Use pyplot to render the graph
        plt.imshow(graph)
        plt.show()
