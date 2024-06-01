import numpy as np

from src.viz import Visualization

from .base import BaseGraph


class Bubbles(BaseGraph):
    viz: Visualization

    def __init__(self, viz: Visualization):
        self.viz = viz

    def draw(self, time_position: int) -> np.ndarray:
        """Draw the graph for a given time frame."""

        palletes = [
            ([200, 0, 0, 255], [0, 0, 255, 255]),  # Red and Blue
            ([0, 255, 0, 255], [255, 255, 0, 255]),  # Greeen and yellow
            ([255, 0, 255, 255], [0, 255, 255, 255]),  # Pink and Cyan
        ]

        # Cycle color palletes and positions based on the current beat
        t = time_position / self.viz.fps
        current_beat = self.viz.audio.bpm * t / 60
        # current_beat_section = current_beat // 96
        current_beat_section = current_beat // 48
        pallete_index = int(current_beat_section % len(palletes))
        pallete = palletes[pallete_index]

        # Get the energy of the audio at the given time position between 20 and 200 Hz
        bass_amp = self.viz.audio.get_energy(time_position, 20, 200)
        bass_amp = np.interp(bass_amp, [0, 1], [0, 255])

        # min_r = 150
        min_r = 120
        max_r = 350

        if self.viz.use_cache and (
            (graph := self.viz.cache.get_graph_cache_item(time_position)) is not None
        ):
            pass
        else:
            graph = np.zeros((self.viz.size, self.viz.size, 4), dtype=np.uint8)

            # Scale the angle_step based on the pallete index between 1 and .25 descending
            angle_step = np.interp(pallete_index, [0, len(palletes) - 1], [1, 0.25])

            wave = self.viz.audio.get_spectrogram_slice(time_position)
            # Scale the wave to the range [-1, 1]
            wave = np.interp(wave, (wave.min(), wave.max()), (-1, 1))
            for t in [-1, 1]:
                # for time_position in range(0, 180, 1):
                for angle_position in np.arange(0, 180, angle_step):
                    index = int(np.interp(angle_position, [0, 180], [0, len(wave) - 1]))
                    r = np.interp(wave[index], [-1, 1], [min_r, max_r])
                    x = r * np.sin(angle_position) * t
                    y = r * np.cos(angle_position)

                    # x and y can be negative, so we need to add the size to them to make them positive
                    x += self.viz.size / 2
                    y += self.viz.size / 2

                    # Draw the vertex if r is greater than min_r
                    if r > min_r:
                        """Sine function ranges from -1 to 1, so we scale it to range. This has the effect of increasing
                        the size of the circles as they move towards the center but then decrease as they move away from
                        the center, like a wave."""
                        min_px, max_px = 3, 9  # Minimum and maximum pixel sizes
                        scaled_r = ((r - min_r) / (max_r - min_r)) * np.pi
                        scaled_r = np.sin(scaled_r) * (max_px - min_px) + min_px
                        scaled_r = np.clip(int(scaled_r), min_px, max_px)

                        # Scale the color by angle_position
                        circle_color = np.array(pallete[0]) * (
                            1 - angle_position / 180
                        ) + np.array(pallete[1]) * (angle_position / 180)
                        # Scale by the bass amplitude
                        circle_color_scale_factor = np.interp(
                            bass_amp, [0, 255], [150, 255]
                        )
                        circle_color = (
                            circle_color * circle_color_scale_factor / 255
                        ).astype(np.uint8)

                        # Draw a circle of radius increased_size at (x, y) with color scaled_color
                        self.draw_circle(
                            graph,
                            int(x),
                            int(y),
                            scaled_r,
                            circle_color,
                            blur=True,
                            blur_radius=scaled_r / 3,
                        )

            # Cache the graph
            self.viz.cache.save_graph_cache_item(time_position, graph)

        # Post process the graph
        graph = self.post_process(graph, time_position, pallete, bass_amp, min_r)
        return graph

    def post_process(
        self,
        graph: np.ndarray,
        time_position: int,
        pallete: list[list[int]],
        bass_amp: int,
        min_r: int,
    ) -> np.ndarray:
        """Post process the graph for a given time frame."""
        # Add a large circle in the center of the graph
        # The value from the get_energy will be between 0 and 1.We need to scale it up to a higher range before
        # converting it to the 255 range because naturally the energy is very low.
        min_mid_freq = 1000
        max_mid_freq = 6000
        mid_freq_energy = self.viz.audio.get_energy(
            time_position, min_mid_freq, max_mid_freq, freq_scale_factor=0.3
        )
        mid_freq_scale_factor = 75
        mid_freq_energy = np.clip(mid_freq_energy * mid_freq_scale_factor, 0, 1)
        mid_freq_energy = np.interp(mid_freq_energy, [0, 1], [0, 255])

        # use the pallete color with the mid_freq_energy to create a fill color
        fill_color = np.clip(
            np.array(pallete[0]) * (1 - mid_freq_energy / 255)
            + np.array(pallete[1]) * (mid_freq_energy / 255),
            0,
            255,
        )
        # Apply gamma correction to darken high color values
        fill_color = self.darken_color(fill_color, 0.7)
        # Reduce the transparency of the fill color by a factor of 1.5 times the bass amplitude
        fill_color[3] = bass_amp * 1.5
        # Scale the radius of the circle based on the high frequency energy
        circle_r_margin = 10
        circle_r_pad = 60
        circle_r = min_r + (mid_freq_energy / 255 - 1) * circle_r_pad
        circle_r = np.clip(circle_r, None, min_r)
        x_center = self.viz.size // 2
        y_center = self.viz.size // 2

        # "Shake" the center circle in if the bass amplitude is greater than 230
        # if bass_amp > 230:
        #    amount = np.interp(bass_amp, [231, 255], [0, 10])
        #    x_center, y_center = self.beat_shake(x_center, y_center, current_beat, amount)

        # Large center circle
        blur_radius = np.clip(circle_r / 3, 0, 10)
        self.draw_circle(
            graph,
            x_center,
            y_center,
            circle_r - circle_r_margin - blur_radius,
            fill_color,
            blur=True,
            blur_radius=blur_radius,
        )

        """TODO: Flash the screen with a white color if the high frequency energy is greater than 200
        use left and right channels to create a stereo effect?
        high_freq_energy = self.audio.get_energy(time_position, 15000, 22050)
        high_freq_energy = np.clip(high_freq_energy * 1000, 0, 1)  # Scale up
        high_freq_energy = np.interp(high_freq_energy, [0, 1], [0, 255])
        self.flash_graph(graph, intensity, time_position, [255, 255, 255, 255])"""
        """TODO: Draw flashing lines between the vertices with greyscale colors based on the high frequency energy
        TODO: Add recursive alternating colors to the center of the circle at 2/3 the radius
        TODO: Finish drawing the flashing lines connecting the vertices
        TODO: If in a certain beat sequence, rotate continuously
        TODO: Move the remainder to a post-processing function"""

        return graph
