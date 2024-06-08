import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from src.audio import Audio
from src.cache import VizCache

from .base import BaseGraph


class Graph3D(BaseGraph):
    """Graph that draws 3D geometric shapes to visualize audio data.
    
    This visualization used matplotlib's 3D plotting capabilities to draw a 3D sphere that has 3d reverse
    stalagmites, where the height of the stalagmite (spike) is determined by the audio data and the placement is
    determined by the time position (along one axis) and frequency of the audio data (along another)."""

    # TODO: Adapt this to use the audio data to draw a 3D geometric shape
    def draw(self, time_position: int, audio: Audio = None, cache: VizCache = None) -> plt.Axes:
        """Draw a geometric shape to the audio visualization."""
        self.create_spike_sphere()
        return
        # 1) Draw a sphere
        # 2) Place a mesh grid overlay on the sphere

        # Create a sphere
        # u = np.linspace(0, 2 * np.pi, 100)
        # v = np.linspace(0, np.pi, 100)
        # x = 10 * np.outer(np.cos(u), np.sin(v))
        # y = 10 * np.outer(np.sin(u), np.sin(v))
        # z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # ax.plot_surface(x, y, z, color="b", alpha=0.0)

        # Create a mesh grid overlay
        # u = np.linspace(0, 2 * np.pi, 100)
        # v = np.linspace(0, np.pi, 100)
        # x = 10 * np.outer(np.cos(u), np.sin(v))
        # y = 10 * np.outer(np.sin(u), np.sin(v))
        # z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

        # audio_data = audio.get_spectrogram_slice(time_position)
        audio_data = np.interp(
            np.linspace(0, audio.spectrogram.shape[0], 100), np.arange(audio.spectrogram.shape[0]), audio.spectrogram[:, time_position]
        )

        # TODO: Implement the mesh grid overlay with the frequency data as the height of the spikes
        # and distributed across the time position axis

        # Normalize audio_data to have a maximum value of 1
        # audio_data /= np.max(audio_data)
        audio_data = np.interp(audio_data, (audio_data.min(), audio_data.max()), (0, 1))

        # Weight the higher end of the frequency spectrum a little bit higher by applying a non-linear function
        audio_data = np.power(audio_data, 2)

        # Create a mesh grid overlay
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 10 * np.outer(np.cos(u), np.sin(v))
        y = 10 * np.outer(np.sin(u), np.sin(v))

        # Use audio_data to affect the z values
        z = 10 * np.outer(np.ones(np.size(u)), np.cos(v)) + audio_data[:, np.newaxis] 


        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(x, y, z, color="g", alpha=0.5)

        # Disable the axis
        ax.axis("off")

        return ax
    
    def fibonacci_sphere(self, samples=1000):
        """Generate a Fibonacci sphere of points."""
        points = []
        phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append((x, y, z))

        return points

    def create_spike_sphere(self):
        """Create a sphere with spikes at the surface."""
        r = self.size / 3 * 2  # Radius of the sphere - 2/3 of the size
        # Create a mesh grid for the sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))

        sphere = np.array([x, y, z])

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(x, y, z, color='b', alpha=0.5)
        # Mesh grid
        ax.plot_wireframe(x, y, z, color='b', alpha=0.5)

        # Create a Fibonacci sphere of points
        points = self.fibonacci_sphere(1000)

        # Define the range of angles for the circle
        theta_range = [np.radians(45), np.radians(75)]  # Inclination angle range in radians
        phi_range = [np.radians(45), np.radians(75)]  # Azimuthal angle range in radians

        # Plot the points on the sphere at the surface of the other sphere
        for i, point in enumerate(points):
            point = np.array(point) * r
            # Calculate the spherical coordinates of the point
            r_spherical, theta, phi = self.cartesian_to_spherical(point[0], point[1], point[2])
            # Check if the point is within the circle
            if theta_range[0] <= theta <= theta_range[1] and phi_range[0] <= phi <= phi_range[1]:
                color = 'b'  # Color for the pool of water
            else:
                color = 'r'
            # Plot at the surface of the sphere every 4 points
            if i % 4 == 0:
                ax.scatter(point[0], point[1], point[2], color=color)
            # Plot the points on the sphere at the surface of the other sphere

        # Disable the axis
        ax.axis("off")

        plt.show()


    # Function to convert Cartesian coordinates to spherical coordinates
    def cartesian_to_spherical(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)  # Inclination angle
        phi = np.arctan2(y, x)  # Azimuthal angle
        return r, theta, phi
