import os
import pickle
import re
import shutil
from typing import Type

import numpy as np
from matplotlib import pyplot as plt

from src.graphs.base import BaseGraph

CACHE_DIR = ".cache/"
GRAPH_CACHE_KEY = "graph"
IMAGE_CACHE_KEY = "image"


class VizCache:
    """File-based cache class to store the a cache used for the visualization of audio data"""

    filename: str = ""
    cache_dir: str = ""
    graph_cache_dir: str = ""
    graph_cache_files: list = []
    img_cache_dir: str = ""
    img_cache_files: list = []

    def __init__(self, filename: str, graph_class: Type[BaseGraph]):
        """Initialize the cache for a given audio file."""
        self.filename = filename
        # Convert the graph_class into a string to use as a subdirectory in the cache directory
        self.cache_dir = f"{CACHE_DIR}{graph_class.__name__}/"
        self.cache_dir = (
            f"{self.cache_dir}{os.path.splitext(os.path.basename(self.filename))[0]}/"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self._init_graph_cache()
        self._init_img_cache()

    def _init_graph_cache(self):
        """Init the graph cache"""
        self._set_graph_cache_dir()
        self._set_graph_cache_files()

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

    def _set_img_cache_dir(self):
        """Set the image cache directory"""
        self.img_cache_dir = f"{self.cache_dir}img/"
        os.makedirs(self.img_cache_dir, exist_ok=True)

    def _set_img_cache_files(self):
        """Set the image cache files in the image cache directory"""
        d = [
            f"{self.img_cache_dir}{f}" for f in os.listdir(self.img_cache_dir)
        ]
        regex_filter = re.compile(r"^(\d+)(\.\w+)?$")
        # Filter and sort by integer value of filename
        d = sorted(
            filter(lambda x: regex_filter.match(os.path.basename(x)), d),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        self.img_cache_files = d

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

    def get_graph_cache_item(self, i: int) -> np.ndarray | plt.Axes:
        """Return the graph cache item for a given index"""
        if not self.graph_cache_file_exists(i):
            return
        if self.graph_cache_files[i].endswith(".npy"):
            return np.load(f"{self.graph_cache_dir}{i}.npy")
        elif self.graph_cache_files[i].endswith(".pickle"):
            with open(f"{self.graph_cache_dir}{i}.pickle", "rb") as f:
                return pickle.load(f)

    def graph_cache_file_exists(self, i: int) -> bool:
        """Check if a graph cache file exists"""
        return i < len(self.graph_cache_files)

    def save_graph_cache_item(self, i: int, graph: np.ndarray | plt.Axes):
        """Save the graph cache to a file so it can be used later"""
        if isinstance(graph, np.ndarray):
            np.save(f"{self.graph_cache_dir}{i}.npy", graph)
        else:
            with open(f"{self.graph_cache_dir}{i}.pickle", "wb") as f:
                pickle.dump(graph, f)

    def get_img_cache_item(self, i: int) -> str:
        """Return the image cache item for a given index"""
        if not self.img_cache_file_exists(i):
            return
        return self.img_cache_files[i]

    def img_cache_file_exists(self, i: int) -> bool:
        """Check if an image cache file exists"""
        return i < len(self.img_cache_files)

    def save_img_cache_item(self, i: int, img: plt.Axes):
        """Save the image cache to a file so it can be used later"""
        img.figure.savefig(f"{self.img_cache_dir}{i}.png", dpi=300, facecolor="black")
        plt.close(img.figure)
