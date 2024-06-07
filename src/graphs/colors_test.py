import numpy as np

from .base import BaseGraph


class ColorsTest(BaseGraph):
    """Graph that tests the color palettes."""
    PALETTES = [
        # Also scale the amount of measures to the total engery
        (((255, 0, 0, 255), (0, 0, 255, 255)), 4),  # Red and Blue
        (((0, 255, 0, 255), (255, 255, 0, 255)), 2),  # Green and Yellow
        # (((255, 0, 255, 255), (0, 255, 255, 255)), 6),  # Magenta and Cyan
        # (((0, 255, 255, 255), (255, 165, 0, 255)), 4),  # Cyan and Orange
        (((255, 20, 147, 255), (57, 255, 20, 255)), 2),  # Deep Pink and Bright Green
        (((255, 255, 0, 255), (255, 0, 255, 255)), 6),  # Yellow and Magenta
    ]

    def draw(self, time_position: int, size: int=720, *args, **kwargs):
        """Draw a simple gradient graph."""
        palette, _ = self.PALETTES[time_position % len(self.PALETTES)]
        angle_step = 1
        gradient = []
        for angle_position in np.arange(0, 180, angle_step):
            # Scale the color by angle_position
            circle_color = self.interpolate_colors(
                palette[0], palette[1], angle_position / 180
            )
            gradient.append(circle_color)
        # Reverse each half of the gradient
        gradient = gradient + gradient[::-1]
        # Render the gradient in a rectangle
        graph = np.zeros((size, size, 4), dtype=np.uint8)
        # Plot the gradient along the x-axis
        division = size // len(gradient)
        for i, color in enumerate(gradient):
            graph[:, i * division:(i + 1) * division] = color
        return graph
