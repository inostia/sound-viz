import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

CACHE_DIR = "cache/"
GRAPH_CACHE_KEY = "graph"
IMAGE_CACHE_KEY = "image"


class VizCache:
    """Cache class to store the a cache used for the visualization of audio data"""

    filename: str = ""
    size: int = 0
    cache: dict = {}
    cache_dir: str = ""
    graph_cache_dir: str = ""
    graph_cache_files: list = []
    img_cache_dir: str = ""
    img_cache_files: list = []

    def __init__(self, filename, size):
        self.filename = filename
        self.size = size
        self._init_cache()

    def _init_cache(self):
        """Set the graph cache directory"""
        self.cache = {}
        self.cache_dir = (
            f"{CACHE_DIR}{os.path.splitext(os.path.basename(self.filename))[0]}/"
        )
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def clear_cache(self):
        """Clear the cache directory"""
        print(f"Clearing cache directory: {self.cache_dir}")
        for filename in os.listdir(self.cache_dir):
            # remove any directories
            if os.path.isdir(f"{self.cache_dir}{filename}"):
                shutil.rmtree(f"{self.cache_dir}{filename}")
            else:
                os.remove(f"{self.cache_dir}{filename}")

    def init_graph_cache(self):
        """Init the graph cache"""
        self._set_graph_cache_dir()
        self._set_graph_cache_files()
        self.cache[GRAPH_CACHE_KEY] = [None] * self.size
        # self.load_graph_cache()

    def _set_graph_cache_dir(self):
        """Set the graph cache directory"""
        self.graph_cache_dir = f"{self.cache_dir}graph/"
        if not os.path.exists(self.graph_cache_dir):
            os.makedirs(self.graph_cache_dir)

    def _set_graph_cache_files(self):
        """Set the graph cache files in the graph cache directory"""
        self.graph_cache_files = os.listdir(self.graph_cache_dir)

    def init_img_cache(self):
        """Init the image cache"""
        self._set_img_cache_dir()
        self._set_img_cache_files()
        self.cache[IMAGE_CACHE_KEY] = [None] * self.size

    def _set_img_cache_dir(self):
        """Set the image cache directory"""
        self.img_cache_dir = f"{self.cache_dir}img/"
        # Clear the image cache directory
        shutil.rmtree(self.img_cache_dir, ignore_errors=True)
        os.makedirs(self.img_cache_dir)

    def _set_img_cache_files(self):
        """Set the image cache files in the image cache directory"""
        self.img_cache_files = os.listdir(self.img_cache_dir)

    def load_graph_cache(self):
        """Load the graph cache from the cache directory"""
        num_cache_files = len(self.graph_cache_files)
        if num_cache_files == 0:
            return
        # Iterate over the cache directory and load each graph cache item
        print(f"Loading {num_cache_files} graph cache items...")
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(self.get_graph_cache_item, i): i
                for i in range(num_cache_files)
            }
            for future in as_completed(futures):
                i = futures[future]
                self.cache[GRAPH_CACHE_KEY][i] = future.result()

    def graph_cache_contains(self, i) -> bool:
        """Check if an item is in the graph cache"""
        return (
            GRAPH_CACHE_KEY in self.cache
            and i < len(self.cache[GRAPH_CACHE_KEY])
            and self.cache[GRAPH_CACHE_KEY][i] is not None
        )

    def graph_cache_file_exists(self, i) -> bool:
        """Check if a graph cache file exists"""
        return f"{i}.npy" in self.graph_cache_files

    def get_graph_cache_item(self, i) -> np.ndarray:
        """Return the graph cache item for a given index"""
        if self.graph_cache_contains(i):
            return self.cache[GRAPH_CACHE_KEY][i]
        if not self.graph_cache_file_exists(i):
            return
        return np.load(f"{self.graph_cache_dir}{i}.npy")

    def save_graph_cache_item(self, i, graph):
        """Save the graph cache to a file so it can be used later"""
        np.save(f"{self.graph_cache_dir}{i}.npy", graph)
        if i >= len(self.cache[GRAPH_CACHE_KEY]):
            self.cache[GRAPH_CACHE_KEY] += [None] * (
                i - len(self.cache[GRAPH_CACHE_KEY]) + 1
            )
        self.cache[GRAPH_CACHE_KEY][i] = graph
