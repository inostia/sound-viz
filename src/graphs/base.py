from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from src.audio import Audio
    from src.cache import VizCache


class BaseGraph(ABC):
    """Base class for all graph types."""

    size: int = 720
    fps: int = 30
    use_cache: bool = False

    def __init__(self, size: int = 720, fps: int = 30, use_cache: bool = False):
        self.size = size
        self.fps = fps
        self.use_cache = use_cache

    def get_graph(self, time_position: int, cache: "VizCache"):
        """Get a graph from the cache or create a new one."""
        if self.use_cache and (
            (graph := cache.get_graph_cache_item(time_position)) is not None
        ):
            # If the graph is in the cache, return it and set the cache hit flag to True
            return graph, True
        # If the graph is not in the cache, create a new graph and return it with the cache hit flag set to False
        graph = np.zeros((self.size, self.size, 4), dtype=np.uint8)
        return graph, False

    # Graph manipulation functions
    @abstractmethod
    def draw(
        self, time_position: int, async_mode: str, audio: "Audio", cache: "VizCache", *args, **kwargs
    ) -> np.ndarray:
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
        self, audio: "Audio", time_position: int, point: tuple[int, int, int], amount: int
    ) -> tuple[int, int]:
        """Shake the x and y coords by a small amount
        
        TODO: This doesn't really represent the physics of a shake, we need to do some more work on this
        before it's ready for prime time."""
        x, y, z = point
        beat = audio.get_beat(time_position, self.fps)

        # Create a random number generator with a seed based on the current beat
        beat_rng = np.random.default_rng(seed=int(beat))

        # Make the scale factor start slow and then accelerate by taking the square root of the fractional part of current_beat
        scale_factor = np.sqrt(beat % 1)

        # Generate a random shake angle for each frame within a beat
        shake_angle = beat_rng.uniform(0, 2 * np.pi)

        # Calculate the x, y, and z shift based on the shake angle and scale factor
        x_shift = scale_factor * amount * np.cos(shake_angle)
        y_shift = scale_factor * amount * np.sin(shake_angle)
        z_shift = scale_factor * amount * np.sin(shake_angle)

        # Update the coordinates with the shake
        x += int(x_shift)
        y += int(y_shift)
        z += int(z_shift)

        return x, y, z

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
        graph[:] = cv2.warpAffine(
            graph, rotation_matrix, (graph.shape[1], graph.shape[0])
        )

    # Color adjustment functions
    def adjust_brightness(self, color: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Apply gamma correction to adjust the brightness of the color"""
        # Separate the color into RGB and alpha components
        rgb, alpha = color[:3], color[3:]

        # Apply gamma correction to the RGB components
        adjusted_rgb = 255 * (rgb / 255) ** gamma

        # Combine the adjusted RGB components with the alpha component
        return np.concatenate([adjusted_rgb, alpha])

    def adjust_transparency(
        self, color: np.ndarray, alpha_factor: float = 1.0
    ) -> np.ndarray:
        """Adjust the transparency of the color by a given factor"""
        # Separate the color into RGB and alpha components
        rgb, old_alpha = color[:3], color[3:]

        # Calculate the new alpha value
        new_alpha = old_alpha * alpha_factor

        # Combine the RGB components with the new alpha value
        return np.concatenate([rgb, new_alpha])

    def set_transparency(self, color: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Set the transparency of the color to a given value"""
        # Separate the color into RGB and alpha components
        rgb = color[:3]

        # Combine the RGB components with the new alpha value
        return np.concatenate([rgb, [alpha]])

    def adjust_saturation(
        self, color: np.ndarray, saturation_factor: float = 1.0
    ) -> np.ndarray:
        """Adjust the saturation of the color by a given factor"""
        # Separate the color into RGB and alpha components
        rgb, alpha = color[:3], color[3:]

        # Convert the RGB color to HSV color space
        hsv_color = self.rgb_to_hsv(rgb)

        # Adjust the saturation of the HSV color
        hsv_color[1] *= saturation_factor

        # Convert the HSV color back to RGB color space
        adjusted_rgb = self.hsv_to_rgb(hsv_color)

        # Combine the adjusted RGB components with the alpha component
        return np.concatenate([adjusted_rgb, alpha])

    def adjust_hue(self, color: np.ndarray, hue_factor: float = 0.0) -> np.ndarray:
        """Adjust the hue of the color by a given factor"""
        # Separate the color into RGB and alpha components
        rgb, alpha = color[:3], color[3:]

        # Convert the RGB color to HSV color space
        hsv_color = self.rgb_to_hsv(rgb)

        # Adjust the hue of the HSV color
        hsv_color[0] += hue_factor
        hsv_color[0] = hsv_color[0] % 360

        # Convert the HSV color back to RGB color space
        adjusted_rgb = self.hsv_to_rgb(hsv_color)

        # Combine the adjusted RGB components with the alpha component
        return np.concatenate([adjusted_rgb, alpha])

    def adjust_contrast(
        self, color: np.ndarray, contrast_factor: float = 1.0
    ) -> np.ndarray:
        """Adjust the contrast of the color by a given factor"""
        # Separate the color into RGB and alpha components
        rgb, alpha = color[:3], color[3:]

        # Convert the RGB color to Lab color space
        lab_color = self.rgb_to_lab(rgb)

        # Adjust the contrast of the Lab color
        lab_color[0] = 128 + contrast_factor * (lab_color[0] - 128)

        # Convert the Lab color back to RGB color space
        adjusted_rgb = self.lab_to_rgb(lab_color)

        # Combine the adjusted RGB components with the alpha component
        return np.concatenate([adjusted_rgb, alpha])

    def adjust_color(self, color: np.ndarray, color_adjustments: dict) -> np.ndarray:
        """Adjust the color of the color by the given adjustments"""
        # Apply the color adjustments to the color
        adjusted_color = color
        color_ops = {
            "brightness": self.adjust_brightness,
            "transparency": self.adjust_transparency,
            "saturation": self.adjust_saturation,
            "hue": self.adjust_hue,
            "contrast": self.adjust_contrast,
        }
        for adjustment, value in color_adjustments.items():
            if value != 0 and adjustment in color_ops:
                adjusted_color = color_ops[adjustment](adjusted_color, value)
        return adjusted_color

    def interpolate_colors(
        self, color1: np.ndarray, color2: np.ndarray, weight: float = 0.5
    ) -> np.ndarray:
        """Interpolate between two colors in the Lab color space."""
        # Save the alpha channel
        alpha = color1[3:]

        # Convert colors to Lab color space
        lab_color1 = self.rgb_to_lab(color1[:3])
        lab_color2 = self.rgb_to_lab(color2[:3])

        # Interpolate in Lab color space
        interpolated_color = lab_color1 * (1 - weight) + lab_color2 * weight

        # Convert back to RGB color space
        rgb_interpolated_color = self.lab_to_rgb(interpolated_color)

        return np.concatenate([rgb_interpolated_color, alpha])

    # Color conversion functions
    def hsv_to_rgb(self, hsv_color) -> np.ndarray:
        """Convert an HSV color to RGB color space."""
        hsv_color = np.array(hsv_color, dtype=np.float32)
        rgb_color = cv2.cvtColor(
            np.array([[hsv_color]], dtype=np.float32), cv2.COLOR_HSV2RGB
        )[0][0]
        return (rgb_color * 255).astype(int)

    def rgb_to_hsv(self, rgb_color) -> np.ndarray:
        """Convert an RGB color to HSV color space."""
        rgb_color = np.array(rgb_color, dtype=np.float32) / 255
        hsv_color = cv2.cvtColor(
            np.array([[rgb_color]], dtype=np.float32), cv2.COLOR_RGB2HSV
        )[0][0]
        return hsv_color

    def rgb_to_lab(self, rgb_color) -> np.ndarray:
        """Convert an RGB color to Lab color space."""
        rgb_color = np.array(rgb_color, dtype=np.float32) / 255
        lab_color = cv2.cvtColor(np.array([[rgb_color]]), cv2.COLOR_RGB2Lab)
        return lab_color[0][0]

    def lab_to_rgb(self, lab_color) -> np.ndarray:
        """Convert a Lab color to RGB color space."""
        rgb_color = cv2.cvtColor(np.array([[lab_color]]), cv2.COLOR_Lab2RGB)
        return (rgb_color[0][0] * 255).astype(int)
