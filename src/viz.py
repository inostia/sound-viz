import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pygame

from src.audio import Audio
from src.cache import VizCache
from src.graphs.base import BaseGraph

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
    graph: BaseGraph

    def __init__(
        self,
        filename: str,
        size: int,
        use_cache: bool = True,
        clear_cache: bool = False,
        bpm: float = None,
        fps: int = 30,
        graph_class: BaseGraph = None,
    ):
        self.size = size
        self.fps = fps
        self.audio = Audio(filename, bpm, fps)
        self.use_cache = use_cache
        self.cache = VizCache(filename, len(self.audio.times))
        if clear_cache:
            self.cache.clear_cache()
        self.cache.init_graph_cache()
        self.cache.init_img_cache()
        self.graph = graph_class(self) if issubclass(graph_class, BaseGraph) else None

    def set_img_dir(self):
        """Set the image directory"""
        self.img_dir = f"images/{time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())}/"
        os.makedirs(self.img_dir)

    def process_frame(self, time_position: int, screen: pygame.Surface = None) -> str:
        """Process a single frame of the visualization."""
        graph = self.graph.draw(time_position)
        # If a screen is provided, render the frame to the screen
        # Otherwise, save the frame to a file
        if screen is not None:
            self.render_frame(screen, graph)
        else:
            return self.save_frame(time_position, graph)

    def save_frame(self, time_position: int, graph: np.ndarray) -> str:
        """Save the graph to a file"""
        image_filename = f"{self.cache.img_cache_dir}{time_position}.png"
        cv2.imwrite(image_filename, graph)
        return image_filename

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
                    num_cache_files = len(self.cache.graph_cache_files)
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
        image_files = self.process_time_frames(async_mode, self.process_frame)

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

    def render_frame(self, screen: pygame.Surface, graph: np.ndarray):
        """Render a single frame"""
        # Convert the graph to a pygame surface
        surface = pygame.surfarray.make_surface(graph)
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

            # Draw the graph
            self.process_frame(
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
