import os

import numpy as np

CACHE_DIR = "cache/"


class Cache:
    """Cache class to store the grid cache"""

    filename: str = ""
    cache_dir: str = ""

    def __init__(self, filename):
        self.filename = filename
        self.set_cache_dir()

    def set_cache_dir(self):
        """Set the grid cache directory"""
        self.cache_dir = f"{CACHE_DIR}{os.path.splitext(os.path.basename(self.filename))[0]}/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_grid_cache_item(self, i):
        """Return the grid cache item for a given index"""
        if not os.path.exists(f"{self.cache_dir}/grid/{i}.npy"):
            return
        grid = np.load(f"{self.cache_dir}/grid/{i}.npy")
        return grid

    def save_grid_cache_item(self, i, grid):
        """Save the grid cache to a file so it can be used later"""
        if not os.path.exists(f"{self.cache_dir}/grid"):
            os.makedirs(f"{self.cache_dir}/grid")
        np.save(f"{self.cache_dir}/grid/{i}.npy", grid)
