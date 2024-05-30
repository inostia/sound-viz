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
from src.cache import Cache

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
    cache: Cache
    fps: float

    def __init__(self, filename: str, size: int):
        self.size = size
        self.audio = Audio(filename)
        self.cache = Cache(filename)
        self.set_img_dir()
        self.set_fps()

    def set_fps(self):
        """Set the frames per second."""
        sample_rate = self.audio.sample_rate  # Get the sample rate of the audio data
        frame_size = self.audio.hop_length  # Get the frame size
        self.fps = sample_rate / frame_size

    def set_img_dir(self):
        """Set the image directory"""
        self.img_dir = f"images/{time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())}/"
        os.makedirs(self.img_dir)

    def draw_circle(self, grid, x_center, y_center, radius, color, blur=False):
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
                temp_grid[:, :, channel] = gaussian_filter(temp_grid[:, :, channel], sigma=radius/3)

        np.add(grid, temp_grid, out=grid, casting="unsafe")

    def draw(self, time_position: int, screen: pygame.Surface = None) -> str | None:
        """Draw the grid for a given time frame.

        Modeled on the following JS code:

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
        }
        """
        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # Get the energy of the audio at the given time position between 20 and 200 Hz
        bass_amp = self.audio.get_energy(time_position, 20, 200)
        bass_amp_max = -1000
        bass_amp_min = -2000
        bass_amp = np.interp(bass_amp, [bass_amp_min, bass_amp_max], [0, 255])
        # Scale the color to bass amplitude from 0 to 255
        alpha = np.interp(bass_amp, [0, 255], [150, 255])
        palletes = [
            ([255, 0, 0], [0, 0, 255]),  # Red and Blue
            ([0, 255, 0], [255, 255, 0]),  # Greeen and yellow
            ([255, 0, 255], [0, 255, 255]),  # Pink and Cyan
        ]
        
        # Cycle color palletes and positions based on the current beat
        t = time_position / self.fps
        current_beat = self.audio.bpm * t // 60
        current_beat_section = current_beat // 96
        current_beat_section = current_beat // 96
        pallete_index = int(current_beat_section % len(palletes))
        pallete = palletes[pallete_index]
        # Scale the angle_step based on the pallete index between 1 and .25 descending
        angle_step = np.interp(pallete_index, [0, len(palletes) - 1], [1, 0.25])

        min_r = 150
        max_r = 350
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
                    color = np.array(pallete[0]) * (1 - angle_position / 180) + np.array(pallete[1]) * (angle_position / 180)

                    # Scale the color by alpha
                    scaled_color = (color * alpha / 255).astype(np.uint8)

                    # Scale the individual pixel size by a factor of r, but limit the size between 1 and 6 pixels
                    # increased_size = ((r - min_r) / (max_r - min_r)) * (6 - 1) + 1
                    # increased_size = np.clip(int(increased_size), 1, 6)

                    # Scale the individual pixel size by a factor of r, but limit the size between 1 and 6 pixels
                    scaled_r = ((r - min_r) / (max_r - min_r)) * np.pi
                    # sine function ranges from -1 to 1, so we scale it to range
                    min_size = 2
                    max_size = 6
                    increased_size = np.sin(scaled_r) * (max_size - min_size) + min_size
                    increased_size = np.clip(int(increased_size), min_size, max_size)

                    # Draw a circle of radius increased_size at (x, y) with color scaled_color
                    self.draw_circle(grid, int(x), int(y), increased_size, scaled_color, True)

        # Cache the grid
        # self.cache.save_grid_cache_item(time_position, grid)

        # "Shake" the grid in a random direction if the bass amplitude is greater than 230
        if bass_amp > 230:
            # Apply a smoothing function to the amount of shaking
            shake = np.interp(bass_amp, [0, 255], [-10, 10])
            # grid = np.roll(grid, np.random.randint(-10, 10), axis=np.random.randint(0, 2))
            grid = np.roll(grid, shake, axis=np.random.randint(0, 2))

        # If a screen is provided, render the frame to the screen
        # Otherwise, save the frame to a file
        if screen is not None:
            self.render_frame(screen, grid)
        else:
            return self.save_frame(time_position, grid)

    def save_frame(self, time_position: int, grid: np.ndarray) -> str:
        """Save the grid to a file"""
        image_filename = f"{self.img_dir}{time_position}.png"
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
            self.draw(int(pygame.time.get_ticks() / 1000 * self.audio.time_index_ratio), screen)

            # Cap the frame rate
            t = pygame.time.get_ticks()
            while t - getTicksLastFrame < 1000 / self.fps:
                t = pygame.time.get_ticks()
            getTicksLastFrame = t

        # Done! Time to quit.
        pygame.quit()

    def process_time_frames(self, process_async: bool, func: callable, *args, **kwargs):
        """Iterate over each time frame and pass the spectrogram slice to the given function"""
        n = len(self.audio.times)
        collection = [None] * n  # Initialize a list of the same length as self.times
        if not process_async:
            print("Processing time frames...")
            for i in range(n):
                collection[i] = func(i, *args, **kwargs)
                print(f"Processed frame {i+1} of {n}")
        else:
            # with ThreadPoolExecutor(max_workers=60) as executor:
            with ProcessPoolExecutor(max_workers=12) as executor:
                # with ThreadPoolExecutor(max_workers=12) as executor:
                print("Processing time frames...")
                futures = {executor.submit(func, i, *args, **kwargs): i for i in range(n)}
                for future in as_completed(futures):
                    i = futures[future]
                    res = future.result()
                    collection[i] = res
                    print(f"Processed frame {i+1} of {n}")
        return collection

    def create_video(self, video_async: bool = False):
        """Create a video from the images generated for each time frame and add the original audio."""
        # Gen_file and save an image for each time frame
        image_files = self.process_time_frames(video_async, self.draw)

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
        video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*"avc1"), self.fps, (width, height))

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