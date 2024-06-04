import numpy as np

from src.audio import Audio
from src.cache import VizCache

from .base import BaseGraph


class Bubbles(BaseGraph):
    """A graph that draws circles in a wave pattern and pulses a center circle based on the frequency amplitudes."""
    PALETTES = [
        (((255, 0, 0, 255), (0, 0, 255, 255)), 1),  # Red and Blue
        (((0, 255, 0, 255), (255, 255, 0, 255)), .5),  # Green and Yellow
        (((255, 0, 255, 255), (0, 255, 255, 255)), 2.5),  # Magenta and Cyan
        (((0, 255, 255, 255), (255, 165, 0, 255)), 2),  # Cyan and Orange
        (((255, 20, 147, 255), (57, 255, 20, 255)), 1),  # Deep Pink and Bright Green
        (((255, 255, 0, 255), (255, 0, 255, 255)), 1),  # Yellow and Magenta
    ]
    ANGLE_STEPS: list = [0.25, 0.625, 0.333]  # Repetitions of the wave pattern in each half of the circle
    ROTATE_CHANCE: float = 0.33  # Chance to rotate the graph


    def draw(
        self,
        time_position: int,
        audio: Audio,
        cache: VizCache,
        size: int = 720,
        fps: int = 30,
        use_cache: bool = False,
    ) -> np.ndarray:
        """Draw the graph for a given time frame."""
        
        """TODO: Flash the screen with a white color if the high frequency energy is greater than 200
        use left and right channels to create a stereo effect?
        high_freq_energy = audio.get_energy(time_position, 15000, 22050)
        high_freq_energy = np.clip(high_freq_energy * 1000, 0, 1)  # Scale up
        high_freq_energy = np.interp(high_freq_energy, [0, 1], [0, 255])
        self.flash_graph(graph, intensity, time_position, [255, 255, 255, 255])"""
        
        # Sooner:
        # TODO: Split the circle into 3(?) sections: 0-60, 60-120, 120-180

        # Later:
        # TODO: Draw flashing lines between the vertices with greyscale colors based on the high frequency energy
        # TODO: Add recursive alternating colors to the center of the circle at 2/3 the radius
        # TODO: Finish drawing the flashing lines connecting the vertices

        # Cycle color palettes and positions based on the current beat
        current_beat = audio.get_beat(time_position, fps)

        # Parse the time signature 
        beats, unit = audio.parse_time_signature()

        if not self.PALETTES:
            raise ValueError("No palettes defined for the graph")

        total_beats = 0
        palette_id = 0
        palette = None

        while total_beats <= current_beat:
            palette, measures = self.PALETTES[palette_id % len(self.PALETTES)]
            total_beats += measures * beats
            if total_beats > current_beat:
                break
            palette_id += 1

        if palette is None:
            raise ValueError("Could not find a palette for the current beat")

        # Get the energy of the audio at the given time position between 20 and 200 Hz
        bass_amp = audio.get_energy(time_position, 20, 200)
        bass_amp = np.interp(bass_amp, [0, 1], [0, 255])

        min_r = 120
        max_r = 350

        if use_cache and (
            (graph := cache.get_graph_cache_item(time_position)) is not None
        ):
            pass
        else:
            graph = np.zeros((size, size, 4), dtype=np.uint8)

            # Draw a large circle in the center of the graph that pulses based on the mid frequency energy
            min_mid_freq = 1000
            max_mid_freq = 6000
            mid_freq_energy = audio.get_energy(
                time_position, min_mid_freq, max_mid_freq, freq_scale_factor=0.3
            )
            mid_freq_scale_factor = 75
            mid_freq_energy = np.clip(mid_freq_energy * mid_freq_scale_factor, 0, 1)
            mid_freq_energy = np.interp(mid_freq_energy, [0, 1], [0, 255])

            # Scale the radius of the circle based on the high frequency energy
            center_circle = {
                "margin": 10,
                "pad": 60,
                # Radius of the center circle
                "r": np.clip(min_r + (mid_freq_energy / 255 - 1) * 60, None, min_r),
            }

            rng = np.random.default_rng(seed=palette_id)
            # Do we want to rotate the graph?
            if palette_id != 0 and rng.random() < self.ROTATE_CHANCE:
                # Aways rotate in the opposite direction as the previous rotation
                rotate_direction = -1 if palette_id % 2 == 0 else 1
                # rotation_rate = rng.uniform(0.5, 0.75) * rotate_direction
                rotation_rate = rng.uniform(0.5, 1.25) * rotate_direction
                # Calculate the rotation angle based on the time position
                rotation_angle = rotation_rate * time_position
                graph = self.rotate_graph(graph, rotation_angle)

            # Add a large circle in the center of the graph
            # The value from the get_energy will be between 0 and 1.We need to scale it up to a higher range before
            # converting it to the 255 range because naturally the energy is very low.
            palette = self.PALETTES[palette_id % len(self.PALETTES)][0]
            fill_color = self.interpolate_color(palette[0], palette[1], mid_freq_energy / 255 * 0.7)
            # Apply gamma correction to darken high color values
            fill_color = self.adjust_brightness(fill_color, 0.3)
            # Set the transparency of the fill color by a factor of 1.5 times the bass amplitude
            fill_color = self.set_transparency(fill_color, np.clip(bass_amp * 1.5, 0, 255))

            x_center = size // 2
            y_center = size // 2

            # Uncomment to "shake" the center circle in if the bass amplitude is greater than 230
            # if bass_amp > 230:
            #    amount = np.interp(bass_amp, [231, 255], [0, 10])
            #    x_center, y_center = self.beat_shake(x_center, y_center, current_beat, amount)

            # Large center circle
            blur_radius = np.clip(center_circle["r"] / 3, 0, 10)
            self.draw_circle(
                graph,
                x_center,
                y_center,
                center_circle["r"] - center_circle["margin"] - blur_radius,
                fill_color,
                blur=True,
                blur_radius=blur_radius,
            )

            # TODO: Scale min_r of the outer circles based on the center r
            # min_r = min_r - center_circle["r"] + center_circle["margin"]

            # Scale the angle_step based on the palette index between 1 and .25 descending
            # angle_step_range = [1, 0.25]
            # angle_step = np.interp(palette_i, [0, len(palettes) - 1], angle_step_range)
            angle_step = self.ANGLE_STEPS[(palette_id % len(self.PALETTES)) % len(self.ANGLE_STEPS)]
            wave = audio.get_spectrogram_slice(time_position)
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
                    x += size / 2
                    y += size / 2

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
                        # circle_color = np.array(palette[0]) * (
                        #     1 - angle_position / 180
                        # ) + np.array(palette[1]) * (angle_position / 180)
                        # Scale the color by angle_position
                        circle_color = self.interpolate_color(palette[0], palette[1], angle_position / 180)
                        # Scale brightness by the bass amplitude
                        gamma = np.interp(bass_amp, [0, 255], [0.8, 1.2])
                        circle_color = self.adjust_brightness(circle_color, gamma)

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
            cache.save_graph_cache_item(time_position, graph, memory_safe=True)

        return graph
