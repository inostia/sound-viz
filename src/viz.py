import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pygame
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from src.audio import Audio
from src.cache import VizCache

"""This is ported from the following p5.js code:

var song
var img
var fft
var particles = []

function preload() {
  song = loadSound('everglow.mp3')
  img = loadImage('bg.jpg')
}

function setup() {
  createCanvas(windowWidth, windowHeight);
  angleMode(DEGREES)
  imageMode(CENTER)
  rectMode(CENTER)
  fft = new p5.FFT(0.3)

  img.filter(BLUR, 12)

  noLoop()
}

function draw() {
  background(0)
  

  translate(width / 2, height / 2)

  fft.analyze()
  amp = fft.getEnergy(20, 200)

  push()
  if (amp > 230) {
    rotate(random(-0.5, 0.5))
  }

  image(img, 0, 0, width + 100, height + 100)
  pop()

  var alpha = map(amp, 0, 255, 180, 150)
  fill(0, alpha)
  noStroke()
  rect(0, 0, width, height)

  
  stroke(255)
  strokeWeight(3)
  noFill()

  var wave = fft.waveform()

  for (var t = -1; t <= 1; t += 2) {
    beginShape()
    for (var i = 0; i <= 180; i += 0.5) {
      var index = floor(map(i, 0, 180, 0, wave.length - 1))
  
      var r = map(wave[index], -1, 1, 150, 350)
      
      var x = r * sin(i) * t
      var y = r * cos(i)
      vertex(x, y)
    }
    endShape()
  }
  
  var p = new Particle()
  particles.push(p)


  for (var i = particles.length - 1; i >= 0; i--) {
    if (!particles[i].edges()) {
      particles[i].update(amp > 230)
      particles[i].show()
    } else {
      particles.splice(i, 1)
    }
    
  }
  
}

function mouseClicked() {
  if (song.isPlaying()) {
    song.pause()
    noLoop()
  } else {
    song.play()
    loop()
  }
}

class Particle {
  constructor() {
    this.pos = p5.Vector.random2D().mult(250)
    this.vel = createVector(0, 0)
    this.acc = this.pos.copy().mult(random(0.0001, 0.00001))

    this.w = random(3, 5)

    this.color = [random(200, 255), random(200, 255), random(200, 255),]
  }
  update(cond) {
    this.vel.add(this.acc)
    this.pos.add(this.vel)
    if (cond) {
      this.pos.add(this.vel)
      this.pos.add(this.vel)
      this.pos.add(this.vel)
    }
  }
  edges() {
    if (this.pos.x < -width / 2 || this.pos.x > width / 2 || this.pos.y < -height / 2 || this.pos.y > height / 2) {
      return true
    } else {
      return false
    }
  }
  show() {
    noStroke()
    fill(this.color)
    ellipse(this.pos.x, this.pos.y, this.w)
  }
}:

"""


class Visualization:
    """Class for visualizing audio data."""

    size: int
    audio: Audio
    img_dir: str
    use_cache: bool
    cache: VizCache
    fps: float

    def __init__(
        self,
        filename: str,
        size: int,
        use_cache: bool = True,
        clear_cache: bool = False,
        bpm: float = None,
        fps: int = 30,
    ):
        self.size = size
        self.fps = fps
        self.audio = Audio(filename, bpm, fps)
        self.use_cache = use_cache
        self.cache = VizCache(filename, len(self.audio.times))
        if clear_cache:
            self.cache.clear_cache()
        self.cache.init_grid_cache()
        self.cache.init_img_cache()

    def set_img_dir(self):
        """Set the image directory"""
        self.img_dir = f"images/{time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())}/"
        os.makedirs(self.img_dir)

    def draw_circle(
        self,
        grid: np.ndarray,
        x_center: int,
        y_center: int,
        radius: int,
        color: np.ndarray,
        blur: bool = False,
    ):
        """Use the midpoint circle algorithm to draw a filled circle on the grid. Apply a blur effect if needed."""
        temp_grid = np.zeros_like(grid)
        x = radius
        y = 0
        err = 0

        while x >= y:
            for i in range(int(x_center - x), int(x_center + x)):
                temp_grid[int(y_center + y), i, :] = color
                temp_grid[int(y_center - y), i, :] = color

            for i in range(int(x_center - y), int(x_center + y)):
                temp_grid[int(y_center + x), i, :] = color
                temp_grid[int(y_center - x), i, :] = color

            if err <= 0:
                y += 1
                err += 2 * y + 1
            if err > 0:
                x -= 1
                err -= 2 * x + 1

        if blur:
            for channel in range(temp_grid.shape[2]):
                if channel < 3:  # Don't blur the alpha channel
                    temp_grid[:, :, channel] = gaussian_filter(
                        temp_grid[:, :, channel], sigma=radius / 3
                    )

        np.add(grid, temp_grid, out=grid, casting="unsafe")

    def beat_shake(
        self, x_center: int, y_center: int, beat: float, amount: int = 0
    ) -> tuple[int, int]:
        """Shake the x and y coords by a small amount"""
        # Create a random number generator with a seed based on the current beat
        beat_rng = np.random.default_rng(seed=int(beat))

        # Get the fractional part of current_beat
        current_beat_frac = beat - int(beat)

        # Make the scale factor start slow and then accelerate by taking the square root of the fractional part of current_beat
        scale_factor = np.sqrt(current_beat_frac)

        # Generate a random shake angle for each frame within a beat
        shake_angle = beat_rng.uniform(0, 2 * np.pi)

        # Calculate the x and y shift based on the shake angle and scale factor
        x_shift = scale_factor * amount * np.cos(shake_angle)
        y_shift = scale_factor * amount * np.sin(shake_angle)

        # Update the center of the circle
        x_center += int(x_shift)
        y_center += int(y_shift)

        return x_center, y_center

    def flash_grid(
        self, grid: np.ndarray, intensity: int, time_position: int, color: list[int]
    ):
        # WIP
        pass

    def draw(self, time_position: int, screen: pygame.Surface = None) -> str | None:
        """Draw the grid for a given time frame."""

        palletes = [
            ([200, 0, 0, 255], [0, 0, 255, 255]),  # Red and Blue
            ([0, 255, 0, 255], [255, 255, 0, 255]),  # Greeen and yellow
            ([255, 0, 255, 255], [0, 255, 255, 255]),  # Pink and Cyan
        ]

        # Cycle color palletes and positions based on the current beat
        t = time_position / self.fps
        current_beat = self.audio.bpm * t / 60
        # current_beat_section = current_beat // 96
        current_beat_section = current_beat // 48
        pallete_index = int(current_beat_section % len(palletes))
        pallete = palletes[pallete_index]

        # Get the energy of the audio at the given time position between 20 and 200 Hz
        bass_amp = self.audio.get_energy(time_position, 20, 200)
        bass_amp = np.interp(bass_amp, [0, 1], [0, 255])

        # min_r = 150
        min_r = 120
        max_r = 350

        if self.use_cache and (
            (grid := self.cache.get_grid_cache_item(time_position)) is not None
        ):
            pass
        else:
            grid = np.zeros((self.size, self.size, 4), dtype=np.uint8)

            # Scale the color to bass amplitude from 0 to 255
            alpha = np.interp(bass_amp, [0, 255], [150, 255])
            # Scale the angle_step based on the pallete index between 1 and .25 descending
            angle_step = np.interp(pallete_index, [0, len(palletes) - 1], [1, 0.25])

            wave = self.audio.get_spectrogram_slice(time_position)
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
                    x += self.size / 2
                    y += self.size / 2

                    # Draw the vertex if r is greater than min_r
                    if r > min_r:
                        # Original color (white)
                        # color = np.array([255, 255, 255])
                        # Scale the color between red and blue based on the angle_position
                        # color = np.array([255, 0, 0]) * (1 - angle_position / 180) + np.array([0, 0, 255]) * (angle_position / 180)
                        color = np.array(pallete[0]) * (
                            1 - angle_position / 180
                        ) + np.array(pallete[1]) * (angle_position / 180)

                        # Scale the color by alpha
                        scaled_color = (color * alpha / 255).astype(np.uint8)
                        # Scale the individual pixel size by a factor of r
                        scaled_r = ((r - min_r) / (max_r - min_r)) * np.pi
                        # Limit the size
                        min_px = 3
                        max_px = 9
                        # Sine function ranges from -1 to 1, so we scale it to range
                        # This has the effect of increasing the size of the circles as they move towards the center
                        # but then decrease as they move away from the center, like a wave
                        increased_size = np.sin(scaled_r) * (max_px - min_px) + min_px
                        increased_size = np.clip(int(increased_size), min_px, max_px)

                        # Draw a circle of radius increased_size at (x, y) with color scaled_color
                        self.draw_circle(
                            grid, int(x), int(y), increased_size, scaled_color, True
                        )

            # Cache the grid
            self.cache.save_grid_cache_item(time_position, grid)

        # The value from the get_energy will be between 0 and 1.
        # We need to scale it up to a higher range before converting it to the 255 range
        # because naturally the energy is very low.
        min_mid_freq = 1000
        max_mid_freq = 6000
        mid_freq_energy = self.audio.get_energy(
            time_position, min_mid_freq, max_mid_freq, freq_scale_factor=0.2
        )
        mid_freq_scale_factor = 75
        mid_freq_energy = np.clip(mid_freq_energy * mid_freq_scale_factor, 0, 1)
        mid_freq_energy = np.interp(mid_freq_energy, [0, 1], [0, 255])

        # use the pallete color with the mid_freq_energy to create a fill color
        fill_color = np.clip(
            np.array(pallete[1]) * (1 - mid_freq_energy / 255)
            + np.array(pallete[0]) * (mid_freq_energy / 255),
            0,
            255,
        )
        # Reduce the transparency of the fill color by a factor of 1.5 times the bass amplitude
        fill_color[3] = bass_amp * 1.5
        # Scale the radius of the circle based on the high frequency energy
        circle_r_margin, circle_r_pad = (10, 60)
        circle_r = min_r + (mid_freq_energy / 255 - 1) * circle_r_pad
        circle_r = np.clip(circle_r, None, min_r)
        x_center = self.size // 2
        y_center = self.size // 2

        # "Shake" the center circle in if the bass amplitude is greater than 230
        # if bass_amp > 230:
        #    amount = np.interp(bass_amp, [231, 255], [0, 10])
        #    x_center, y_center = self.beat_shake(x_center, y_center, current_beat, amount)

        # Large center circle
        self.draw_circle(
            grid, x_center, y_center, circle_r - circle_r_margin, fill_color, True
        )

        """TODO: Flash the screen with a white color if the high frequency energy is greater than 200
        use left and right channels to create a stereo effect?
        high_freq_energy = self.audio.get_energy(time_position, 15000, 22050)
        high_freq_energy = np.clip(high_freq_energy * 1000, 0, 1)  # Scale up
        high_freq_energy = np.interp(high_freq_energy, [0, 1], [0, 255])
        self.flash_grid(grid, intensity, time_position, [255, 255, 255, 255])"""

        """TODO: Draw flashing lines between the vertices with greyscale colors based on the high frequency energy
        TODO: Add recursive alternating colors to the center of the circle at 2/3 the radius
        TODO: Finish drawing the flashing lines connecting the vertices
        TODO: If in a certain beat sequence, rotate continuously
        TODO: Move the remainder to a post-processing function"""

        # If a screen is provided, render the frame to the screen
        # Otherwise, save the frame to a file
        if screen is not None:
            self.render_frame(screen, grid)
        else:
            return self.save_frame(time_position, grid)

    def save_frame(self, time_position: int, grid: np.ndarray) -> str:
        """Save the grid to a file"""
        image_filename = f"{self.cache.img_cache_dir}{time_position}.png"
        cv2.imwrite(image_filename, grid)
        return image_filename

    def render_frame(self, screen: pygame.Surface, grid: np.ndarray):
        """Render a single frame"""
        # Convert the grid to a pygame surface
        surface = pygame.surfarray.make_surface(grid)
        surface = pygame.transform.scale(surface, (self.size, self.size))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    def render(self, screen: pygame.Surface):
        t = pygame.time.get_ticks()
        getTicksLastFrame = t
        # pygame.mixer.music.load(self.audio.filename)
        # pygame.mixer.music.play(0)

        # Run until the user asks to quit
        running = True
        while running:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw the grid
            self.draw(
                int(pygame.time.get_ticks() / 1000 * self.audio.time_index_ratio),
                screen,
            )

            # Cap the frame rate
            t = pygame.time.get_ticks()
            while t - getTicksLastFrame < 1000 / self.fps:
                t = pygame.time.get_ticks()
            getTicksLastFrame = t

        # Done! Time to quit.
        pygame.quit()

    def process_time_frames(self, async_mode: str, func: callable, *args, **kwargs):
        """Iterate over each time frame and pass the spectrogram slice to the given function"""
        n = len(self.audio.times)
        collection = [None] * n  # Initialize a list of the same length as self.times
        if async_mode == "on":
            max_processes, max_threads = 12, 12
            with ProcessPoolExecutor(
                max_workers=max_processes
            ) as process_exector, ThreadPoolExecutor(
                max_workers=max_threads
            ) as thread_executor:
                if self.use_cache:
                    # Assume the cache files are in order
                    num_cache_files = len(self.cache.grid_cache_files)
                    thread_futures = {
                        thread_executor.submit(func, i, *args, **kwargs): i
                        for i in range(num_cache_files)
                    }
                    if len(thread_futures) > 0:
                        print(
                            f"Processing {len(thread_futures)} cache files using {max_threads} threads..."
                        )
                    for future in as_completed(thread_futures):
                        i = thread_futures[future]
                        collection[i] = future.result()
                        print(f"Processed frame {i+1} of {num_cache_files} from cache")
                    # Release the thread futures
                    thread_executor.shutdown(wait=True)
                else:
                    num_cache_files = 0
                # Process the remaining frames
                process_futures = {
                    process_exector.submit(func, i, *args, **kwargs): i
                    for i in range(num_cache_files, n)
                }
                if len(process_futures) > 0:
                    print(
                        f"Processing {len(process_futures)} frames using {max_processes} processes..."
                    )
                for future in as_completed(process_futures):
                    i = process_futures[future]
                    collection[i] = future.result()
                    print(f"Processed frame {i+1} of {n} from audio")
        else:
            print("Processing time frames synchronously...")
            for i in range(n):
                collection[i] = func(i, *args, **kwargs)
                print(f"Processed frame {i+1} of {n}")
        return collection

    def create_video(self, async_mode: str = "off"):
        """Create a video from the images generated for each time frame and add the original audio."""
        # Gen_file and save an image for each time frame
        image_files = self.process_time_frames(async_mode, self.draw)

        # Get the size of and name image
        img = cv2.imread(image_files[0])
        height, width, layers = img.shape
        basename = os.path.basename(self.audio.filename)
        video_filename, output_filename = (
            f"output/{basename}_video.mp4",
            f"output/{basename}_output.mp4",
        )

        # Create a VideoWriter object
        print(f"Creating video from {len(image_files)} images...")
        video = cv2.VideoWriter(
            video_filename, cv2.VideoWriter_fourcc(*"avc1"), self.fps, (width, height)
        )

        # Write each image tage_filedeo
        for image_file in image_files:
            video.write(cv2.imread(image_file))

        # Release the VideoWriter
        video.release()

        # Optionally, remove the image files
        # for image_file in image_files:
        #     os.remove(image_file)

        # Add the original audio to the video
        command = [
            "ffmpeg",
            "-i",
            video_filename,
            "-i",
            self.audio.filename,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            output_filename,
            "-y",
        ]
        subprocess.run(command, check=True)
        return video_filename, output_filename
