import os
import shutil

import numpy as np

CACHE_DIR = ".cache/"
GRAPH_CACHE_KEY = "graph"
IMAGE_CACHE_KEY = "image"


class VizCache:
    """Cache class to store the a cache used for the visualization of audio data"""

    filename: str = ""
    length: int = 0
    cache: dict = {}
    cache_dir: str = ""
    graph_cache_dir: str = ""
    graph_cache_files: list = []
    img_cache_dir: str = ""
    img_cache_files: list = []

    def __init__(self, filename: str, length: int):
        """Initialize the cache with a given filename and length."""
        self.filename = filename
        self.length = length
        self.cache = {}
        self.cache_dir = (
            f"{CACHE_DIR}{os.path.splitext(os.path.basename(self.filename))[0]}/"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self._init_graph_cache()
        self._init_img_cache()

    def _init_graph_cache(self):
        """Init the graph cache"""
        self._set_graph_cache_dir()
        self._set_graph_cache_files()
        self.cache[GRAPH_CACHE_KEY] = [None] * self.length

    def _set_graph_cache_dir(self):
        """Set the graph cache directory"""
        self.graph_cache_dir = f"{self.cache_dir}graph/"
        os.makedirs(self.graph_cache_dir, exist_ok=True)

    def _set_graph_cache_files(self):
        """Set the graph cache files in the graph cache directory"""
        # Sort by integer value of filename
        self.graph_cache_files = sorted(
            os.listdir(self.graph_cache_dir), key=lambda x: int(os.path.splitext(x)[0])
        )

    def _init_img_cache(self):
        """Init the image cache"""
        self._set_img_cache_dir()
        self._set_img_cache_files()
        self.cache[IMAGE_CACHE_KEY] = [None] * self.length

    def _set_img_cache_dir(self):
        """Set the image cache directory"""
        self.img_cache_dir = f"{self.cache_dir}img/"
        os.makedirs(self.img_cache_dir, exist_ok=True)

    def _set_img_cache_files(self):
        """Set the image cache files in the image cache directory"""
        self.img_cache_files = os.listdir(self.img_cache_dir)

    # Public methods
    def clear_cache(self):
        """Clear the cache directory"""
        print(f"Clearing cache directory: {self.cache_dir}")
        for filename in os.listdir(self.cache_dir):
            # remove any directories
            if os.path.isdir(f"{self.cache_dir}{filename}"):
                shutil.rmtree(f"{self.cache_dir}{filename}")
            else:
                os.remove(f"{self.cache_dir}{filename}")
    
    def clear_graph_cache(self):
        """Clear the graph cache directory"""
        print(f"Clearing graph cache directory: {self.graph_cache_dir}")
        for filename in os.listdir(self.graph_cache_dir):
            os.remove(f"{self.graph_cache_dir}{filename}")

    def clear_img_cache(self):
        """Clear the image cache directory"""
        print(f"Clearing image cache directory: {self.img_cache_dir}")
        for filename in os.listdir(self.img_cache_dir):
            os.remove(f"{self.img_cache_dir}{filename}")

    def get_graph_cache_item(self, i: int) -> np.ndarray:
        """Return the graph cache item for a given index"""
        if self.graph_cache_contains(i):
            return self.cache[GRAPH_CACHE_KEY][i]
        if not self.graph_cache_file_exists(i):
            return
        return np.load(f"{self.graph_cache_dir}{i}.npy")

    def graph_cache_contains(self, i: int) -> bool:
        """Check if an item is in the graph cache"""
        return (
            GRAPH_CACHE_KEY in self.cache
            and i < len(self.cache[GRAPH_CACHE_KEY])
            and self.cache[GRAPH_CACHE_KEY][i] is not None
        )

    def graph_cache_file_exists(self, i: int) -> bool:
        """Check if a graph cache file exists"""
        return f"{i}.npy" in self.graph_cache_files

    def save_graph_cache_item(
        self, i: int, graph: np.ndarray, memory_safe: bool = False
    ):
        """Save the graph cache to a file so it can be used later"""
        np.save(f"{self.graph_cache_dir}{i}.npy", graph)
        if not memory_safe:
            self.set_graph_cache_item(i, graph)

    def set_graph_cache_item(self, i: int, graph: np.ndarray):
        """Set the graph cache item for a given index"""
        if i >= len(self.cache[GRAPH_CACHE_KEY]):
            self.cache[GRAPH_CACHE_KEY] += [None] * (
                i - len(self.cache[GRAPH_CACHE_KEY]) + 1
            )
        self.cache[GRAPH_CACHE_KEY][i] = graph
