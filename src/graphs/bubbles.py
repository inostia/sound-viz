import numpy as np

from src.audio import Audio
from src.cache import VizCache

from .base import BaseGraph


class Bubbles(BaseGraph):
    """A graph that draws circles in a wave pattern and pulses a center circle based on the frequency amplitudes."""

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
        # TODO: Draw flashing lines between the vertices with greyscale colors based on the high frequency energy
        # TODO: Add recursive alternating colors to the center of the circle at 2/3 the radius
        # TODO: Finish drawing the flashing lines connecting the vertices
        # TODO: Split the circle into 3(?) sections: 0-60, 60-120, 120-180
        # TODO: IF IN A CERTAIN BEAT SEQUENCE (24 BEATS), ROTATE CONTINUOUSLY
        # TODO: Take the time signature as an argument and perform calculations based on that


        PALETTES = {
            ((255, 0, 0, 255), (0, 0, 255, 255)): 12,  # Red and Blue
            ((0, 255, 0, 255), (255, 255, 0, 255)): 6,  # Green and Yellow
            ((255, 0, 255, 255), (0, 255, 255, 255)): 30,  # Magenta and Cyan
            ((0, 255, 255, 255), (255, 165, 0, 255)): 24,  # Cyan and Orange
            ((255, 20, 147, 255), (57, 255, 20, 255)): 12,  # Deep Pink and Bright Green
            ((255, 255, 0, 255), (255, 0, 255, 255)): 12,  # Yellow and Magenta
        }
        ANGLE_STEPS = [0.25, 0.625, 0.333]  # Repetitions of the wave pattern in each half of the circle

        # Cycle color palettes and positions based on the current beat
        t = time_position / fps
        current_beat = audio.bpm * t / 60

        palette, palette_i, total_duration = None, 0, 0
        palettes_items = list(PALETTES.items())
        total_beats = sum(duration for _, duration in palettes_items)

        # Calculate the effective current beat by taking modulus with total beats
        # This will make the current beat cycle back to 0 after it exceeds total beats
        effective_current_beat = current_beat % total_beats

        # Find the current palette
        for i, (p, duration) in enumerate(palettes_items):
            total_duration += duration
            if effective_current_beat < total_duration:
                palette = p
                palette_i = i
                break

        if palette is None:
            raise ValueError("Could not find a palette for the current beat")

        # Get the energy of the audio at the given time position between 20 and 200 Hz
        bass_amp = audio.get_energy(time_position, 20, 200)
        bass_amp = np.interp(bass_amp, [0, 1], [0, 255])

        # min_r = 150
        min_r = 120
        max_r = 350

        if use_cache and (
            (graph := cache.get_graph_cache_item(time_position)) is not None
        ):
            pass
        else:
            graph = np.zeros((size, size, 4), dtype=np.uint8)

            # Scale the angle_step based on the palette index between 1 and .25 descending
            # angle_step_range = [1, 0.25]
            # angle_step = np.interp(palette_i, [0, len(palettes) - 1], angle_step_range)
            angle_step = ANGLE_STEPS[palette_i % len(ANGLE_STEPS)]
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

        # Post process the graph
        graph = self.post_process(graph, time_position, palette, bass_amp, min_r, audio, size)
        return graph

    def post_process(
        self,
        graph: np.ndarray,
        time_position: int,
        palette: list[list[int]],
        bass_amp: int,
        min_r: int,
        audio: Audio,
        size: int = 720,
    ) -> np.ndarray:
        """Post process the graph for a given time frame."""
        # Add a large circle in the center of the graph
        # The value from the get_energy will be between 0 and 1.We need to scale it up to a higher range before
        # converting it to the 255 range because naturally the energy is very low.
        min_mid_freq = 1000
        max_mid_freq = 6000
        mid_freq_energy = audio.get_energy(
            time_position, min_mid_freq, max_mid_freq, freq_scale_factor=0.3
        )
        mid_freq_scale_factor = 75
        mid_freq_energy = np.clip(mid_freq_energy * mid_freq_scale_factor, 0, 1)
        mid_freq_energy = np.interp(mid_freq_energy, [0, 1], [0, 255])

        # use the palette color with the mid_freq_energy to create a fill color
        # fill_color = np.clip(
        #     np.array(palette[0]) * (1 - mid_freq_energy / 255)
        #     + np.array(palette[1]) * (mid_freq_energy / 255),
        #     0,
        #     255,
        # )
        # fill_color = self.interpolate_color(palette[0], palette[1], mid_freq_energy / 255)
        # Adjust the weight of mid_freq_energy to make the first color in the palette more dominant
        # fill_color = np.clip(
        #     np.array(palette[0]) * (1 - mid_freq_energy / 255 * 0.5)
        #     + np.array(palette[1]) * (mid_freq_energy / 255),
        #     0,
        #     255,
        # )
        fill_color = self.interpolate_color(palette[0], palette[1], mid_freq_energy / 255 * 0.7)
        # Apply gamma correction to darken high color values
        # fill_color = self.adjust_brightness(fill_color, 0.7)
        # Set the transparency of the fill color by a factor of 1.5 times the bass amplitude
        fill_color = self.set_transparency(fill_color, np.clip(bass_amp * 1.5, 0, 255))
        # Scale the radius of the circle based on the high frequency energy
        circle_r_margin = 10
        circle_r_pad = 60
        circle_r = min_r + (mid_freq_energy / 255 - 1) * circle_r_pad
        circle_r = np.clip(circle_r, None, min_r)
        x_center = size // 2
        y_center = size // 2

        # Uncomment to "shake" the center circle in if the bass amplitude is greater than 230
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

        return graph
