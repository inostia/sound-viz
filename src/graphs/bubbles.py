import numpy as np

from src.audio import Audio
from src.cache import VizCache

from .base import BaseGraph


class Bubbles(BaseGraph):
    """A graph that draws circles in a wave pattern and pulses a center circle based on the frequency amplitudes."""

    PALETTES = [
        # TODO: Use random palettes
        # Also scale the amount of measures to the total engery
        (((255, 0, 0, 255), (0, 0, 255, 255)), 4),  # Red and Blue
        (((0, 255, 0, 255), (255, 255, 0, 255)), 2),  # Green and Yellow
        (((255, 0, 255, 255), (0, 255, 255, 255)), 6),  # Magenta and Cyan
        # (((0, 255, 255, 255), (255, 165, 0, 255)), 4),  # Cyan and Orange
        (((255, 20, 147, 255), (57, 255, 20, 255)), 2),  # Deep Pink and Bright Green
        (((255, 255, 0, 255), (255, 0, 255, 255)), 6),  # Yellow and Magenta
        # (("#3BE600", "#DB00BC"), 1),  # Green and Pink
        # (("#100085", "#DBBE00"), 1),  # Blue and Yellow
    ]
    ANGLE_STEPS: list = [
        0.25,
        0.625,
        0.8,
    ]  # Repetitions of the wave pattern in each half of the circle
    ROTATE_CHANCE: float = 0.5  # Chance to rotate the graph
    # SPLIT_CHANCE: tuple[int, float] = (24, 0.5)  # After n measures, chance to split the circle into 3 sections
    SPLIT_CHANCE: tuple[int, float] = (
        3,
        0.5,
    )  # After n measures, chance to split the circle into 3 sections

    def draw(
        self,
        time_position: int,
        size: int = 720,
        audio: Audio = None,
        cache: VizCache = None,
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

        # Use the initialize graph method to get the graph, palette_id, palette, and bass_amp
        graph, cache_hit, palette_id, palette, bass_amp, mid_amp, rng = (
            self.initialize_graph(time_position, audio, cache, size, use_cache, fps)
        )
        if cache_hit:
            return graph

        min_r, max_r = 120, 350
        self.draw_outer_circles(
            time_position,
            audio,
            graph,
            size,
            palette_id,
            palette,
            bass_amp,
            min_r,
            max_r,
        )
        self.draw_center_circle(graph, size, palette, bass_amp, mid_amp, min_r)
        self.rotate_graph(graph, palette_id, time_position, rng)

        # Cache the graph
        cache.save_graph_cache_item(time_position, graph, memory_safe=True)
        return graph

    def initialize_graph(
        self,
        time_position: int,
        audio: Audio,
        cache: VizCache,
        size: int,
        use_cache: bool,
        fps: int,
    ):
        graph, cache_hit = self.get_graph(time_position, cache, size, use_cache)
        if cache_hit:
            return graph, cache_hit, None, None, None, None, None
        current_beat = audio.get_beat(time_position, fps)
        beats, unit = audio.parse_time_signature()
        palette_id, palette = self.get_palette(current_beat, beats, unit)
        rng = np.random.default_rng(seed=palette_id)
        # Which measure are we on?
        measure = current_beat // beats * (4 / unit)
        # TODO: Split the circle into 3 sections
        if measure > self.SPLIT_CHANCE[0]:
            # Split the palette up - draw a circle with the x,y at 0,0 so that it's only in a single quadrant
            # Then flip/move it to the correct quadrant
            split_chance = rng.random() < self.SPLIT_CHANCE[1]
        bass_amp = self.get_bass_amplitude(time_position, audio)
        mid_amp = self.get_mid_amplitude(time_position, audio)
        return graph, cache_hit, palette_id, palette, bass_amp, mid_amp, rng

    def measures_beats(self, measures: int, beats: int, unit: int):
        """Get the number of beats in n measures."""
        return measures * beats * (4 / unit)

    def get_palette(self, current_beat: int, beats: int, unit: int):
        """Get the palette for the current beat."""
        if not self.PALETTES:
            raise ValueError("No palettes defined for the graph")
        total_beats = 0
        palette_id = 0
        palette = None
        palettes = [palette for palette, _ in self.PALETTES]
        palette_measures = [measures for _, measures in self.PALETTES]
        palette_beats = sum(
            [
                self.measures_beats(measures, beats, unit)
                for measures in palette_measures
            ]
        )
        while total_beats <= current_beat:
            # TODO: Choose a measure based on the total energy of the song
            rng = np.random.default_rng(seed=palette_id)
            if current_beat < palette_beats:
                palette, measures = self.PALETTES[palette_id % len(self.PALETTES)]
            else:
                palette = rng.choice(palettes)
                # Chance to flip the palette
                if rng.random() < 0.5:
                    palette = palette[::-1]
                measures = rng.choice(list(set(palette_measures)))
            total_beats += self.measures_beats(measures, beats, unit)
            if total_beats > current_beat:
                break
            palette_id += 1
        if palette is None:
            raise ValueError("Could not find a palette for the current beat")
        if isinstance(palette[0], str):
            # Convert the hex to rgb
            palette = [
                tuple(int(p[i : i + 2], 16) for i in (1, 3, 5)) + (255,)
                for p in palette
            ] 
        return palette_id, palette

    def rotate_graph(
        self,
        graph: np.ndarray,
        palette_id: int,
        time_position: int,
        rng: np.random.Generator,
    ):
        """Chance to rotate the graph based on the palette_id and time_position."""
        # Do we want to rotate the graph?
        if palette_id != 0 and rng.random() < self.ROTATE_CHANCE:
            # Aways rotate in the opposite direction as the previous rotation
            rotate_direction = -1 if palette_id % 2 == 0 else 1
            # rotation_rate = rng.uniform(0.5, 0.75) * rotate_direction
            rotation_rate = rng.uniform(0.5, 1.25) * rotate_direction
            # Calculate the rotation angle based on the time position
            rotation_angle = rotation_rate * time_position
            super().rotate_graph(graph, rotation_angle)

    def get_bass_amplitude(self, time_position: int, audio: Audio):
        """Get the bass amplitude for the current time position."""
        bass_amplitude = audio.get_energy(time_position, 20, 200)
        bass_amplitude = np.interp(bass_amplitude, [0, 1.5], [0, 255])
        return bass_amplitude

    def get_mid_amplitude(self, time_position: int, audio: Audio, multiplier: int = 75):
        """Get the mid frequency amplitude for the current time position."""
        min_mid_freq = 1000
        max_mid_freq = 6000
        mid_amplitude = audio.get_energy(
            time_position, min_mid_freq, max_mid_freq, freq_scale_factor=0.3
        )
        mid_amplitude = np.clip(mid_amplitude * multiplier, 0, 1)
        mid_amplitude = np.interp(mid_amplitude, [0, 1], [0, 255])
        return mid_amplitude

    def get_graph(
        self, time_position: int, cache: VizCache, size: int, use_cache: bool
    ):
        """Get a graph from the cache or create a new one."""
        if use_cache and (
            (graph := cache.get_graph_cache_item(time_position)) is not None
        ):
            # If the graph is in the cache, return it and set the cache hit flag to True
            return graph, True
        # If the graph is not in the cache, create a new graph and return it with the cache hit flag set to False
        graph = np.zeros((size, size, 4), dtype=np.uint8)
        return graph, False

    def draw_outer_circles(
        self,
        time_position: int,
        audio: Audio,
        graph: np.ndarray,
        size: int,
        palette_id: int,
        palette: tuple,
        bass_amp: float,
        min_r: int,
        max_r: int,
    ):
        """Draw the outer circles in a wave pattern based on the audio wave."""
        # TODO: Scale min_r of the outer circles based on the center r
        # min_r = min_r - center_circle["r"] + center_circle["margin"]

        # Scale the angle_step based on the palette index between 1 and .25 descending
        angle_step = self.ANGLE_STEPS[
            (palette_id % len(self.PALETTES)) % len(self.ANGLE_STEPS)
        ]
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
                    circle_color = self.interpolate_colors(
                        palette[0], palette[1], angle_position / 180
                    )
                    # Scale brightness by the bass amplitude
                    # gamma = np.interp(bass_amp, [0, 255], [0.8, 1.2])
                    # circle_color = self.adjust_brightness(circle_color, gamma)

                    # Scale transparency of the circles by the bass amplitude
                    transparency = np.interp(bass_amp, [0, 255], [0.2, 1])
                    circle_color = self.adjust_transparency(circle_color, transparency)

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

    def draw_center_circle(
        self,
        graph: np.ndarray,
        size: int,
        palette: tuple,
        bass_amp: float,
        mid_amp: float,
        min_r: int = 120,
    ):
        # Scale the radius of the circle based on the high frequency energy
        margin = 10
        # Normalize mid_amp to a range between -1 and 0
        normalized_mid_amp = mid_amp / 255 - 1
        # Scale the normalized mid_amp by a factor of 60
        scaled_mid_amp = normalized_mid_amp * 60
        # Calculate the radius based on min_r and the scaled mid_amp - ensure the radius doesn't exceed min_r
        r = np.clip(
            min_r + scaled_mid_amp, None, min_r
        )  # Radius of the center circle, weighted by mid_amp

        # Add a large circle in the center of the graph
        color_weight = np.interp(np.clip(mid_amp, 0, 255), [0, 255], [1, 0])
        fill_color = self.interpolate_colors(
            palette[0], palette[1], color_weight
        )
        # Saturate the color
        # fill_color = self.adjust_saturation(fill_color, 1)
        # Darken high color values with the inverse of the mid_amp
        # fill_color = self.adjust_brightness(fill_color, np.interp(mid_amp, [0, 255], [0.3, 1]))
        # Set the transparency of the fill color by a factor of 1.5 times the bass amplitude
        # fill_color = self.set_transparency(fill_color, np.clip(bass_amp * 1.5, 0, 255))
        transparency = np.interp(bass_amp, [0, 255], [0.2, 1])
        fill_color = self.adjust_transparency(fill_color, transparency)

        x_center = size // 2
        y_center = size // 2

        # Uncomment to "shake" the center circle in if the bass amplitude is greater than 230
        # if bass_amp > 230:
        #    amount = np.interp(bass_amp, [231, 255], [0, 10])
        #    x_center, y_center = self.beat_shake(x_center, y_center, current_beat, amount)

        # Large center circle
        blur_radius = np.clip(r / 3, 0, 10)

        # TODO: Add texture to the center circle by adding an argument to draw_circle that takes a texture function
        self.draw_circle(
            graph,
            x_center,
            y_center,
            r - margin - blur_radius,
            fill_color,
            blur=True,
            blur_radius=blur_radius,
        )
