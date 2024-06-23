import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.audio import Audio
from src.cache import VizCache, handle_redis_errors
from src.viz import DPI

from .base import BaseGraph

TEST_CAMERA = False


class Graph3D(BaseGraph):
    """Graph that draws 3D geometric shapes to visualize audio data.

    This visualization used matplotlib's 3D plotting capabilities to draw a 3D sphere that has 3d reverse
    stalagmites, where the height of the stalagmite (spike) is determined by the audio data and the placement is
    determined by the time position (along one axis) and frequency of the audio data (along another)."""

    def draw(
        self,
        time_position: int,
        async_mode: str,
        audio: Audio = None,
        cache: VizCache = None,
        *args,
        **kwargs,
    ) -> plt.Axes:
        """Draw a geometric shape to the audio visualization."""
        # TODO: Fix the cache
        if self.use_cache and (cached_img := cache.get_img_cache_item(time_position)):
            return cached_img

        fig, ax = self.create_figure(time_position, async_mode)

        # Get the spectrum data for the current time position
        spectrum_data = self.get_sphere_spectrum(time_position, audio)
        points = self.fibonacci_sphere(len(spectrum_data))
        num_points = len(points)

        # Separate into bands - define the number of frequency bands and the size of the gaps between them
        num_bands = 9
        gap_size = 0
        band_index = 0
        _, band_indices = self.get_band_sizes(spectrum_data, num_bands, gap_size)
        # Calculate the amplitudes by band
        band_amplitudes = self.get_amplitudes(spectrum_data, band_indices)

        # Get the brightness of the points based on the mean amplitude
        sphere_brightness = self.get_brightness(time_position, audio, cache)
        sizes = np.zeros(num_points)
        colors = np.zeros((num_points, 4))

        # Initialize the X, Y, Z coordinates
        X, Y, Z = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)
        min_r = self.size / 2
        max_r = min_r * 2

        # Get the camera position
        elev_angle, azim_angle = self.get_rotation(time_position)
        camera_position = self.get_camera_position(elev_angle, azim_angle, max_r * 15)

        if TEST_CAMERA:
            self.test_camera(camera_position, min_r, ax)
            # ax.axis("off")
            # return fig

        # Use vectorized operations to speed up the calculations iterate over bands
        for band_index, (start_index, end_index) in enumerate(band_indices):
            # Skip any points in the last band
            if band_index >= num_bands - 1:
                continue

            band_amplitude = band_amplitudes[band_index]

            # Calculate the radius of the band
            band_size = end_index - start_index
            band_r = min_r + band_size / len(spectrum_data) * (max_r - min_r)

            # Calculate the amplitude of the band
            amplitude = band_amplitude
            adjusted_r = band_r * (1 + amplitude)
            adjusted_r = np.clip(adjusted_r, min_r, max_r)

            # Reduce the point by a factor of the band index to create a 3D effect
            adjusted_r *= 1 - 0.07 * band_index
            # Add some displacement to the points to create a 3D effect
            displacement_constant = band_index ** np.cbrt(band_index) * 0.33
            # Calculate the displacement based on the amplitude
            mid_point = num_bands // 2
            if band_index < mid_point:
                displacement_factor = -0.1
                displacement_y = amplitude * displacement_factor * displacement_constant
                points[start_index:end_index] += [0, displacement_y, 0]
            else:
                displacement_factor = 0.1
                displacement_y = amplitude * displacement_factor * displacement_constant
                points[start_index:end_index] -= [0, displacement_y, 0]

            # Adjust the position of the points
            points[start_index:end_index] *= adjusted_r

            # Calculate the distance of each point from the origin
            distances = self.calculate_distances(points[start_index:end_index], camera_position)

            # Normalize the distances to [0,1]
            distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

            # Vectorized calculation of sizes based on distances
            distance_size_factor = 1.2
            min_size = 0.3
            sizes = np.append(sizes, min_size + distances * distance_size_factor)

            rgba = np.array([0, sphere_brightness, 0, 1])
            colors = np.vstack([colors, np.tile(rgba, (distances.shape[0], 1))])

            # Plot the points - assuming X, Y, Z are lists that are extended with points coordinates
            X = np.append(X, points[start_index:end_index, 0])
            Y = np.append(Y, points[start_index:end_index, 1])
            Z = np.append(Z, points[start_index:end_index, 2])

        # Plot the points with the adjusted colors
        ax.scatter(X, Y, Z, c=colors, s=sizes)

        ax.axis("off")
        # TODO: Fix the cache
        # cache.save_graph_cache_item(time_position, fig)
        return fig

    def test_camera(
        self, camera_position: tuple[int, int, int], min_r: int, ax: plt.Axes
    ):
        """Plot a test dot at the camera position with the size set to the distance from 0, 0, 0."""
        camera_dist = np.linalg.norm(camera_position)
        ax.scatter(*camera_position, c="red", s=camera_dist)

        # Define the vertices of the cube
        vertices = (
            np.array(
                [
                    [-1, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, 1, 1],
                ]
            )
            * min_r
            * 0.5
        )

        # Define the faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[3], vertices[2]],
            [vertices[4], vertices[5], vertices[7], vertices[6]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[2], vertices[6], vertices[4]],
            [vertices[1], vertices[3], vertices[7], vertices[5]],
        ]

        # Create a Poly3DCollection object where each face is a separate color
        cube = Poly3DCollection(
            faces,
            facecolors=["white", "green", "blue", "yellow", "orange", "purple"],
            edgecolors="k",
        )

        # Add the collection to the axes
        ax.add_collection3d(cube)

    def get_amplitudes(self, spectrum_data: np.ndarray, band_indices: list[tuple[int, int]]) -> list[float]:
        """Calculate the amplitude of each band based on the spectrum data."""
        amplitudes = []
        for start_index, end_index in band_indices:
            band_spectrum_data = spectrum_data[start_index:end_index]
            amplitude_scale = 1.2
            amplitude = np.mean(band_spectrum_data) * amplitude_scale
            amplitudes.append(amplitude)
        return amplitudes

    def get_band_sizes(
        self, spectrum_data: np.ndarray, num_bands: int, gap_size: int
    ) -> list[int]:
        """Calculate the size of each band using a cube root scale based on the spectrum data."""
        # TODO: Dynamic gap size based on the amplitude of the spectrum data
        # Calculate the total size available for the bands
        total_size = len(spectrum_data) - gap_size * (num_bands - 1)

        # Harmonic series 
        series_sum = sum([1 / i for i in range(1, num_bands + 1)])

        # Calculate the size of the first band
        first_band_size = (total_size / series_sum)

        # Calculate the size of each band
        band_sizes = [int(first_band_size / i) for i in range(1, num_bands + 1)]

        # Calculate the start and end indices of each band
        band_indices = []
        start_index = 0
        for band_size in band_sizes:
            end_index = start_index + band_size
            band_indices.append((start_index, end_index))
            start_index = end_index + gap_size

        return band_sizes, band_indices

    def get_rotation(self, time_position: int):
        """Get the rotation angles for the 3D plot. The angles are determined by sine waves that oscillate over time."""
        seconds_per_rotation = 45  # Decrease for a faster rotation
        frames_per_rotation = seconds_per_rotation * self.fps
        elev_wave = np.sin(np.linspace(0, 2 * np.pi, frames_per_rotation))
        azim_wave = np.sin(np.linspace(0, 4 * np.pi, frames_per_rotation))

        # Get the rotation angles from the sine waves
        elev_angle = elev_wave[time_position % len(elev_wave)] * 360
        azim_angle = azim_wave[time_position % len(azim_wave)] * 360

        # Apply a low-frequency oscillator (LFO) to the rotation angles. The LFO is a sine wave that completes a half
        # cycle over the period of frames_per_rotation. This results in a slow oscillation of the rotation angles,
        # creating a more dynamic visual effect.
        lfo_frequency = 0.5  # The frequency of the LFO in Hz
        # Modify lfo_frequency to adjust the speed of the oscillation by a factor of -1 to 1 based on the band index mod
        lfo = np.sin(
            np.linspace(0, 2 * np.pi, frames_per_rotation) * lfo_frequency
        )
        slow_factor = lfo[time_position % len(lfo)]
        elev_angle *= slow_factor
        azim_angle *= slow_factor

        return elev_angle, azim_angle

    def create_figure(
        self, time_position: int, async_mode: str = "off"
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a figure and axes for the graph."""
        if async_mode == "on":
            # This import is necessary to enter a detached head mode when running in a multi-process environment
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        else:
            import matplotlib.pyplot as plt
        # Set the background color to black
        plt.rcParams["figure.facecolor"] = "black"
        plt.rcParams["axes.facecolor"] = "black"
        # Create a 3D plot
        width = self.size / DPI
        height = self.size / DPI

        # fig = plt.figure(figsize=(width, height))
        fig = plt.figure(figsize=(width, height), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        ax.set_zmargin(0)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-self.size, self.size)
        ax.set_ylim(-self.size, self.size)
        ax.set_zlim(-self.size, self.size)

        elev_angle, azim_angle = self.get_rotation(time_position)
        ax.view_init(elev=elev_angle, azim=azim_angle)

        return fig, ax

    @handle_redis_errors
    def get_brightness(
        self, time_position: int, audio: Audio, cache: VizCache
    ) -> float:
        """Calculate the brightness of a point based on the amplitude."""

        # TODO: Calculate a gradient of brightness per frequency band

        def sigmoid(x):
            """Logistic sigmoid function."""
            return 1 / (1 + np.exp(-x))

        # Create a bandpass filter to isolate the mid frequencies
        hp_cutoff, lp_cutoff = 500, 1500
        hp_order, lp_order = 9, 4

        # Check the cache for the spectrogram and time series
        hp_lp_key = f"{audio.filename}:{audio.fps}:{hp_cutoff}:{lp_cutoff}:{hp_order}:{lp_order}:brightness_spectrogram"
        if cache.redis.exists(hp_lp_key):
            spectrogram = pickle.loads(cache.redis.get(hp_lp_key))
        else:
            spectrogram, time_series = audio.highpass_filter(hp_cutoff, order=hp_order)
            spectrogram, _ = audio.lowpass_filter(
                lp_cutoff, order=lp_order, time_series=time_series
            )
            cache.redis.set(hp_lp_key, pickle.dumps(spectrogram))
        mean_amplitudes = np.mean(spectrogram, axis=0)
        max_mean = np.max(mean_amplitudes)
        real_min = -74.34712432871528  # Empirical min value from clip3
        min_mean = np.clip(np.min(mean_amplitudes), real_min, max_mean - 1)
        high_freq_amplitude = mean_amplitudes[time_position]

        # Normalize the amplitude
        brightness_min = 0.0
        brightness_max = 1.0
        high_freq_amplitude = np.interp(
            high_freq_amplitude,
            [min_mean, max_mean],
            [brightness_min, brightness_max],
        )

        high_freq_amplitude = sigmoid(high_freq_amplitude * 10 - 5)

        return high_freq_amplitude

    def get_sphere_spectrum(self, time_position: int, audio: Audio) -> np.ndarray:
        """Process the spectrum data."""
        # TODO: Process in stereo and split the spectrum data into left and right channels

        return np.interp(
            audio.get_spectrogram_slice(time_position),
            [np.min(audio.spectrogram), np.max(audio.spectrogram)],
            [0, 1],
        )

    def fibonacci_sphere(self, samples=1000) -> np.ndarray:
        """Generate a Fibonacci sphere of points."""
        # TODO: Instead of a Fibonacci sphere, use a geodesic sphere or a cube sphere
        points = []
        phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def cartesian_to_spherical(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)  # Inclination angle
        phi = np.arctan2(y, x)  # Azimuthal angle
        return r, theta, phi

    def get_camera_position(self, elev_angle=0, azim_angle=0, r=1):
        """Calculate the camera position in Cartesian coordinates."""
        theta = np.radians(90 - elev_angle)
        phi = np.radians(azim_angle)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return x, y, z

    def calculate_distances(self, points, camera_position=(0, 0, 0)) -> np.ndarray:
        """Calculate the distance of each point from the origin."""
        cx, cy, cz = camera_position
        px, py, pz = points.T
        distances = np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)
        distances = 1 - distances
        return distances
