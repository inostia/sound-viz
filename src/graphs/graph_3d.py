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
        if async_mode == "on":
            # This import is necessary to enter a detached head mode when running in a multi-process environment
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        else:
            import matplotlib.pyplot as plt

        init_plt(plt)

        # Create a 3D plot
        dpi = 100  # dots per inch
        width = self.size / dpi  # calculate weight in inches
        height = self.size / dpi * 9 / 16  # calculate height in inches

        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        ax.set_zmargin(0)
        ax.set_box_aspect([1, 1, 1])

        # TODO: FIX THE CACHE
        # if self.use_cache:
        # cached_graph = cache.get_graph_cache_item(time_position)
        # if cached_graph is not None:
        #     return cached_graph

        # Make a rotating effect by changing the view angle and modifying the elevation and azimuth by sine waves of
        # different frequencies
        # Calculate the number of frames for a full rotation
        # frames_per_rotation = 10 * self.fps
        frames_per_rotation = 50 * self.fps
        elev_wave = np.sin(np.linspace(0, 2 * np.pi, frames_per_rotation))
        azim_wave = np.sin(np.linspace(0, 4 * np.pi, frames_per_rotation))

        # Get the rotation angles from the sine waves
        elev_angle = elev_wave[time_position % len(elev_wave)] * 360
        azim_angle = azim_wave[time_position % len(azim_wave)] * 360

        # Apply a slower sine wave to the rotation angles
        slow_wave = np.sin(np.linspace(0, np.pi, frames_per_rotation))
        slow_factor = slow_wave[time_position % len(slow_wave)]
        elev_angle *= slow_factor
        azim_angle *= slow_factor

        ax.view_init(elev=elev_angle, azim=azim_angle)

        # Get the audio data at the time position
        spectrum_data = audio.get_spectrogram_slice(time_position)

        # V1
        # # Convert spectrum_data from decibel to linear scale
        # # spectrum_data = librosa.db_to_amplitude(spectrum_data)
        # # Subtract the mean from the spectrum data
        # spectrum_data = spectrum_data - np.mean(spectrum_data)
        # # Divide by the standard deviation
        # spectrum_data = spectrum_data / np.std(spectrum_data)
        # # Apply a non-linear transformation to exaggerate the differences
        # spectrum_data = np.sign(spectrum_data) * np.power(spectrum_data, 2)
        
        # V2
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Subtract the mean from the spectrum data
        spectrum_data = spectrum_data - np.mean(spectrum_data)
        # Divide by the standard deviation
        spectrum_data = spectrum_data / np.std(spectrum_data)
        # Apply the sigmoid function to exaggerate the differences
        spectrum_data = sigmoid(spectrum_data)
        # Normalize the spectrum data to [0, 1]
        spectrum_data = (spectrum_data - spectrum_data.min()) / (spectrum_data.max() - spectrum_data.min())
        
        # Get the points of the sphere
        # TODO: Redistribute the points in clusters across the sphere to make the visualization more interesting
        points = self.fibonacci_sphere(len(spectrum_data))

        # Set the size of the sphere
        r = self.size / 2
        # Create some random directions for the arrows
        directions = np.random.rand(10, 3) - 0.5
        # Scale the brightness of the arrows by the bass amplitude
        bass_energy = audio.get_energy(time_position, 0, 200)
        brightness = np.interp(bass_energy, [0, 1.2], [0.5, 1])
        X = []
        Y = []
        Z = []
        for i, point in enumerate(points):
            # TODO: Instead of straight scaling by the amplitude, do something more akin to physics,
            # like a spring system or a wave system. For example, the amplitude could be the amount of
            # displacement from the equilibrium position. We could calculate the velocity of the point
            # and the acceleration of the point based on the amplitude and the previous amplitudes.
            # Then we could shoot the point in the direction of the acceleration and dampen the velocity
            # based on the velocity and the amplitude.

            amplitude = spectrum_data[i]
            adjusted_r = r * (1 + amplitude)
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
                color=(0, brightness, 0)  # Bright green
            )

        # Convert lists to numpy arrays
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)

        # Calculate the maximum possible radius
        max_radius = np.sqrt(X**2 + Y**2 + Z**2).max()

        # Set the limits of the x, y, and z axes based on the maximum radius
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
        ax.set_zlim(-max_radius, max_radius)

        ax.axis("off")

        # TODO: FIX THE CACHE
        # cache.save_graph_cache_item(time_position, ax)

        return fig

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
