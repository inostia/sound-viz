from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from src.viz import Visualization


class BaseGraph(ABC):
    """Base class for all graph types."""
    viz: 'Visualization'

    def __init__(self, viz: 'Visualization'):
        self.viz = viz

    @abstractmethod
    def draw(self, time_position: int) -> np.ndarray:
        """Draw the graph for a given time frame."""
        pass

    def draw_circle(
        self,
        graph: np.ndarray,
        x_center: int,
        y_center: int,
        radius: int,
        color: np.ndarray,
        blur: bool = False,
        blur_radius: int = 3,
    ):
        """Use the midpoint circle algorithm to draw a filled circle on the graph. Apply a blur effect if needed."""
        temp_graph = np.zeros_like(graph)
        x = radius
        y = 0
        err = 0

        while x >= y:
            for i in range(int(x_center - x), int(x_center + x)):
                temp_graph[int(y_center + y), i, :] = color
                temp_graph[int(y_center - y), i, :] = color

            for i in range(int(x_center - y), int(x_center + y)):
                temp_graph[int(y_center + x), i, :] = color
                temp_graph[int(y_center - x), i, :] = color

            if err <= 0:
                y += 1
                err += 2 * y + 1
            if err > 0:
                x -= 1
                err -= 2 * x + 1

        if blur:
            for channel in range(temp_graph.shape[2]):
                if channel < 3:  # Don't blur the alpha channel
                    temp_graph[:, :, channel] = gaussian_filter(
                        temp_graph[:, :, channel], blur_radius
                    )

        np.add(graph, temp_graph, out=graph, casting="unsafe")

    def beat_shake(
        self, x_center: int, y_center: int, beat: float, amount: int = 0
    ) -> tuple[int, int]:
        """Shake the x and y coords by a small amount"""
        # Create a random number generator with a seed based on the current beat
        beat_rng = np.random.default_rng(seed=int(beat))

        # Get the fractional part of current_beat
        current_beat_frac = beat - int(beat)

        # Make the scale factor start slow and then accelerate by taking the square root of the fractional part of current_beat
        scale_factor = np.sqrt(current_beat_frac)

        # Generate a random shake angle for each frame within a beat
        shake_angle = beat_rng.uniform(0, 2 * np.pi)

        # Calculate the x and y shift based on the shake angle and scale factor
        x_shift = scale_factor * amount * np.cos(shake_angle)
        y_shift = scale_factor * amount * np.sin(shake_angle)

        # Update the center of the circle
        x_center += int(x_shift)
        y_center += int(y_shift)

        return x_center, y_center

    def flash_graph(
        self, graph: np.ndarray, intensity: int, time_position: int, color: list[int]
    ):
        # WIP
        pass

    def darken_color(self, color: np.ndarray, gamma: float = 0.8) -> np.ndarray:
        """Apply gamma correction to darken the color"""
        return 255 * (color / 255) ** gamma
