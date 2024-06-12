import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.audio import Audio
from src.cache import VizCache
from src.viz import init_plt

from .base import BaseGraph


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
        if (
            self.use_cache
            and (cached_img := cache.get_img_cache_item(time_position))
        ):
            return cached_img

        fig, ax = self.create_figure(time_position, async_mode)
        X, Y, Z = [], [], []
        min_r = self.size / 2

        # Create some random directions for the arrows
        directions = np.random.rand(10, 3) - 0.5

        # Get the spectrum data for the current time position
        spectrum_data = self.get_spectrum(time_position, audio)
        points = self.fibonacci_sphere(len(spectrum_data))

        # Separate into bands - define the number of frequency bands and the size of the gaps between them
        num_bands = 60
        gap_size = 5
        band_sizes, band_indices = self.get_band_sizes(
            spectrum_data, num_bands, gap_size
        )

        # Initialize the total size covered so far and the current band index
        total_size = 0
        band_index = 0

        # Get the brightness of the points based on the mean amplitude
        # brightness = self.get_brightness(time_position, audio)
        brightness = 1

        # Iterate over the points
        for point in points:
            """TODO: Instead of straight scaling by the amplitude, do something more akin to physics,
            like a spring system or a wave system. For example, the amplitude could be the amount of
            displacement from the equilibrium position. We could calculate the velocity of the point
            and the acceleration of the point based on the amplitude and the previous amplitudes.
            Then we could shoot the point in the direction of the acceleration and dampen the velocity
            based on the velocity and the amplitude."""
            """TODO: Alternatively, draw a wave in the animation every time
            this method is called, and the wave could be a function of the amplitude."""
            """TODO: Redistribute the points in clusters across the sphere to make the visualization more interesting"""
            """TODO: Some points should shoot outwards if the force is strong enough"""
            """TODO: Imprint the word "Xenotech" on the sphere as a cutout of the amplitude"""

            # Increment the total size
            total_size += 1

            # If the total size exceeds the size of the current band, move to the next band
            if total_size > band_sizes[band_index] + gap_size:
                band_index += 1
                total_size = 0

            # If the total size is within the gap size, skip the point
            if total_size <= gap_size:
                continue

            # Get the start and end indices for the current band
            start_index, end_index = band_indices[band_index]

            # Get the spectrum data for the current band
            band_spectrum_data = spectrum_data[start_index:end_index]

            # Calculate the amplitude for the current band
            amplitude_scale = 1
            amplitude = np.mean(band_spectrum_data) * amplitude_scale

            # Use the normalized amplitude to adjust the radius
            adjusted_r = min_r * (1 + amplitude)
            adjusted_r = np.clip(adjusted_r, min_r, self.size)
            point = point * adjusted_r
            X.append(point[0])
            Y.append(point[1])
            Z.append(point[2])

            # Plot the arrows
            ax.quiver(
                point[0],
                point[1],
                point[2],
                directions[:, 0],
                directions[:, 1],
                directions[:, 2],
                length=0.2,
                color=(0, brightness, 0),  # Bright green
            )
        ax.axis("off")
        # TODO: Fix the cache
        # cache.save_graph_cache_item(time_position, fig)
        return fig

    def get_band_sizes(
        self, spectrum_data: np.ndarray, num_bands: int, gap_size: int
    ) -> list[int]:
        """Calculate the size of each band using a cube root scale based on the spectrum data."""
        # TODO: Dynamic gap size based on the amplitude of the spectrum data
        # Calculate the total size available for the bands
        total_size = len(spectrum_data) - gap_size * (num_bands - 1)

        # Calculate the sum of the series 1, 1/cbrt(2), 1/cbrt(3), ..., 1/cbrt(num_bands)
        # series_sum = sum(1 / np.cbrt(i) for i in range(1, num_bands + 1))
        # Harmonic series
        series_sum = sum(1 / i for i in range(1, num_bands + 1))

        # Calculate the size of the first band
        first_band_size = total_size / series_sum

        # Calculate the size of each band
        band_sizes = [
            round(first_band_size / np.cbrt(i)) for i in range(1, num_bands + 1)
        ]

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
        seconds_per_rotation = 50  # Decrease for a faster rotation
        frames_per_rotation = seconds_per_rotation * self.fps
        elev_wave = np.sin(np.linspace(0, 2 * np.pi, frames_per_rotation))
        azim_wave = np.sin(np.linspace(0, 4 * np.pi, frames_per_rotation))

        # Get the rotation angles from the sine waves
        elev_angle = elev_wave[time_position % len(elev_wave)] * 360
        azim_angle = azim_wave[time_position % len(azim_wave)] * 360

        # Apply a low-frequency oscillator (LFO) to the rotation angles. The LFO is a sine wave that completes a half
        # cycle over the period of frames_per_rotation. This results in a slow oscillation of the rotation angles,
        # creating a more dynamic visual effect.
        lfo = np.sin(np.linspace(0, np.pi, frames_per_rotation))
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
        init_plt(plt)
        # Create a 3D plot
        dpi = 100  # dots per inch
        # width = self.size / dpi  # calculate weight in inches
        # height = self.size / dpi * 5 / 4  # calculate height in inches

        # fig = plt.figure(figsize=(width, height))
        fig = plt.figure(figsize=(self.size / dpi, self.size / dpi))
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
    
    def get_brightness(self, time_position: int, audio: Audio) -> float:
        """Calculate the brightness of a point based on the amplitude."""
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        # Get the mean amplitude of the spectrogram only for the high frequencies
        # high_freq = 10000
        high_freq = 10000
        high_freq_index = audio.get_frequency_index(high_freq)
    
        # Get the spectrum data for the current time position
        spectrum_data = audio.get_spectrogram_slice(time_position)
        spectrum_data = spectrum_data[high_freq_index:]

        # Get the mean amplitude of the high frequencies
        high_freq_amplitude = np.mean(spectrum_data)

        # Get the mean amplitude of all the high frequencies in the audio
        # spectrogram_amplitudes = [np.mean(spect[high_freq_index:]) for spect in audio.spectrogram]
        mean_amplitudes = []
        for spect in audio.spectrogram:
            if len(spect) > high_freq_index:
                mean_amplitudes.append(np.mean(spect[high_freq_index:]))
        mean_amplitudes = np.array(mean_amplitudes)
        # Normalize the amplitude
        brightness_min = 0.5
        brightness_max = 1
        high_freq_amplitude = np.interp(
            high_freq_amplitude,
            [np.min(mean_amplitudes), np.max(mean_amplitudes)],
            [brightness_min, brightness_max],
        )

        # Apply the sigmoid function to exaggerate the differences
        high_freq_amplitude = sigmoid(high_freq_amplitude)

        return high_freq_amplitude




    def get_spectrum(self, time_position: int, audio: Audio) -> np.ndarray:
        """Process the spectrum data."""

        # TODO: Process in stereo and split the spectrum data into left and right channels
        spectrum_data = audio.get_spectrogram_slice(time_position)

        # Subtract the mean from the spectrum data
        # spectrum_data = spectrum_data - np.mean(spectrum_data)
        # # Divide by the standard deviation
        # spectrum_data = spectrum_data / np.std(spectrum_data)
        # Apply the sigmoid function to exaggerate the differences
        # spectrum_data = sigmoid(spectrum_data)

        # Apply a non-linear transformation to exaggerate the differences
        # spectrum_data = np.sign(spectrum_data) * np.power(spectrum_data, 2)

        return np.interp(
            spectrum_data,
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

    def cartesian_to_spherical(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)  # Inclination angle
        phi = np.arctan2(y, x)  # Azimuthal angle
        return r, theta, phi
