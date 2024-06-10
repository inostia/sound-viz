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
        # TODO: FIX THE CACHE
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
        fig, ax = self.create_figure(time_position, async_mode)
        X, Y, Z = [], [], []
        min_r = self.size / 2

        # Create some random directions for the arrows
        directions = np.random.rand(10, 3) - 0.5

        # Scale the brightness of the arrows by the bass amplitude
        bass_energy = audio.get_energy(time_position, 0, 200)
        # brightness = np.interp(bass_energy, [0, 1.2], [0.5, 1])
        brightness = 1


        # Get the spectrum data for the current time position
        spectrum_data = self.get_spectrum(time_position, audio)
        points = self.fibonacci_sphere(len(spectrum_data))

        # Separate into bands - define the number of frequency bands and the size of the gaps between them
        num_bands = 10
        gap_size = 5
        band_size = (len(spectrum_data) - gap_size * (num_bands - 1)) // num_bands
        # TODO: Redistribute the points in clusters across the sphere to make the visualization more interesting
        for i, point in enumerate(points):
            # TODO: Instead of straight scaling by the amplitude, do something more akin to physics,
            # like a spring system or a wave system. For example, the amplitude could be the amount of
            # displacement from the equilibrium position. We could calculate the velocity of the point
            # and the acceleration of the point based on the amplitude and the previous amplitudes.
            # Then we could shoot the point in the direction of the acceleration and dampen the velocity
            # based on the velocity and the amplitude.
            # TODO: Alternatively, draw a wave in the animation every time
            # this method is called, and the wave could be a function of the amplitude.

            # Calculate the band index based on the point's position
            band_index = i // (band_size + gap_size)

            # Skip the points in the gaps
            if i % (band_size + gap_size) < gap_size:
                continue

            # Get the spectrum data for the current band
            band_spectrum_data = spectrum_data[band_index * band_size:(band_index + 1) * band_size]

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
            # point = self.beat_shake(audio, time_position, point, amount=0.3)
            # Plot the arrows
            ax.quiver(
                point[0],
                point[1],
                point[2],
                directions[:, 0],
                directions[:, 1],
                directions[:, 2],
                # length=0.1,
                length=0.2,
                # TODO: Pulse the color with the beat and/or low-frequency or the individual frequency
                color=(0, brightness, 0),  # Bright green
            )

        ax.axis("off")

        # TODO: FIX THE CACHE
        # cache.save_graph_cache_item(time_position, ax)

        return fig

    def get_spectrum(self, time_position: int, audio: Audio) -> np.ndarray:
        """Process the spectrum data."""
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # TODO: Process in stereo and split the spectrum data into left and right channels

        spectrum_data = audio.get_spectrogram_slice(time_position)
        spectrum_data = spectrum_data[50:]  # Remove the sub frequencies
        # Set the high frequencies to a very low number to create a crater in the sphere
        # Subtract the mean from the spectrum data
        # spectrum_data = spectrum_data - np.mean(spectrum_data)
        # # Divide by the standard deviation
        # spectrum_data = spectrum_data / np.std(spectrum_data)
        # Apply the sigmoid function to exaggerate the differences
        # spectrum_data = sigmoid(spectrum_data)

        # Apply a non-linear transformation to exaggerate the differences
        # spectrum_data = np.sign(spectrum_data) * np.power(spectrum_data, 2)

        # Normalize the spectrum data to [0, 1]
        # return (spectrum_data - spectrum_data.min()) / (
        #     spectrum_data.max() - spectrum_data.min()
        # )
        return np.interp(spectrum_data, [np.min(audio.spectrogram), np.max(audio.spectrogram)], [0, 1])

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
