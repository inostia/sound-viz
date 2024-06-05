from abc import ABC, abstractmethod

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class BaseGraph(ABC):
    """Base class for all graph types."""


    @abstractmethod
    def draw(self, time_position: int, *args, **kwargs) -> np.ndarray:
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

    def rotate_graph(self, graph: np.ndarray, angle: float):
        """Rotate the graph in place by a given angle"""
        # Get the center of the graph
        center = (graph.shape[1] // 2, graph.shape[0] // 2)
        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Rotate the graph
        graph[:] = cv2.warpAffine(graph, rotation_matrix, (graph.shape[1], graph.shape[0]))

    def adjust_brightness(self, color: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Apply gamma correction to adjust the brightness of the color"""
        # Separate the color into RGB and alpha components
        rgb, alpha = color[:3], color[3:]

        # Apply gamma correction to the RGB components
        adjusted_rgb = 255 * (rgb / 255) ** gamma

        # Combine the adjusted RGB components with the alpha component
        return np.concatenate([adjusted_rgb, alpha])
    
    def adjust_transparency(self, color: np.ndarray, alpha_factor: float = 1.0) -> np.ndarray:
        """Adjust the transparency of the color by a given factor"""
        # Separate the color into RGB and alpha components
        rgb, old_alpha = color[:3], color[3:]

        # Calculate the new alpha value
        new_alpha = old_alpha * alpha_factor

        # Combine the RGB components with the new alpha value
        return np.concatenate([rgb, [new_alpha]])
    
    def set_transparency(self, color: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Set the transparency of the color to a given value"""
        # Separate the color into RGB and alpha components
        rgb = color[:3]

        # Combine the RGB components with the new alpha value
        return np.concatenate([rgb, [alpha]])

    def rgb_to_lab(self, rgb_color) -> np.ndarray:
        """Convert an RGB color to Lab color space."""
        rgb_color = np.array(rgb_color, dtype=np.float32) / 255
        lab_color = cv2.cvtColor(np.array([[rgb_color]]), cv2.COLOR_RGB2Lab)
        return lab_color[0][0]

    def lab_to_rgb(self, lab_color) -> np.ndarray:
        """Convert a Lab color to RGB color space."""
        rgb_color = cv2.cvtColor(np.array([[lab_color]]), cv2.COLOR_Lab2RGB)
        return (rgb_color[0][0] * 255).astype(int)
    
    def rgba_to_rgb(self, rgba_color) -> np.ndarray:
        """Convert an RGBA color to RGB color space."""
        return rgba_color[:3]

    def interpolate_color(self, color1, color2, weight) -> np.ndarray:
        """Interpolate between two colors in the Lab color space."""
        # Save the alpha channel
        alpha = color1[3:]

        # Convert colors to Lab color space
        lab_color1 = self.rgb_to_lab(self.rgba_to_rgb(color1))
        lab_color2 = self.rgb_to_lab(self.rgba_to_rgb(color2))

        # Interpolate in Lab color space
        interpolated_color = lab_color1 * (1 - weight) + lab_color2 * weight

        # Convert back to RGB color space
        rgb_interpolated_color = self.lab_to_rgb(interpolated_color)

        return np.concatenate([rgb_interpolated_color, alpha])
