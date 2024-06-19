import hashlib
import os
from typing import Type

from src.graphs.base import BaseGraph


def generate_unique_filename(filename) -> str:
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


def parse_graph_class(graph: str) -> Type[BaseGraph]:
    """Load up the graph from the module string"""
    graph_module, graph_class = graph.rsplit(".", 1)
    graph_module = __import__(graph_module, fromlist=[graph_class])
    graph_class = getattr(graph_module, graph_class)
    return graph_class
