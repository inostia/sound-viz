import hashlib
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Type

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

    filename: str
    size: int
    graph_class: Type[BaseGraph]
    bpm: int
    fps: float
    use_cache: bool
    clear_cache: str

    def __init__(
        self,
        filename: str,
        size: int,
        graph_class: Type[BaseGraph],
        bpm: float = None,
        fps: int = 30,
        use_cache: bool = True,
        clear_cache: str = "no",
    ):
        self.filename = filename
        self.size = size
        if not issubclass(graph_class, BaseGraph):
            raise ValueError("graph_class must be a subclass of BaseGraph")
        self.graph_class = graph_class
        self.bpm = bpm
        self.fps = fps
        self.use_cache = use_cache
        self.clear_cache = clear_cache

    def process_frame(self, time_position: int, screen: pygame.Surface = None) -> str:
        """Process a single frame of the visualization."""
        start_time = time.time()
        audio = Audio(self.filename, self.bpm, self.fps)
        cache = VizCache(self.filename, len(audio.times))
        graph = self.graph_class().draw(
            time_position, audio, cache, self.size, self.fps, self.use_cache
        )
        # If a screen is provided, render the frame to the screen
        # Otherwise, save the frame to a file
        if screen is not None:
            self.render_frame(screen, graph)
        else:
            return self.save_frame(
                time_position, graph, cache
            ), time.time() - start_time

    def save_frame(self, time_position: int, graph: np.ndarray, cache: VizCache) -> str:
        """Save the graph to a file"""
        image_filename = f"{cache.img_cache_dir}{time_position}.png"
        cv2.imwrite(image_filename, graph)
        return image_filename

    def process_time_frames(
        self,
        func: callable,
        async_mode: str = "off",
        async_workers: int = 4,
        *args,
        **kwargs,
    ):
        """Iterate over each time frame and pass the spectrogram slice to the given function"""
        audio = Audio(self.filename, self.bpm, self.fps)
        n = len(audio.times)
        cache = VizCache(self.filename, n)
        if self.clear_cache == "all":
            cache.clear_cache()
        elif self.clear_cache == "graph":
            cache.clear_graph_cache()
        elif self.clear_cache == "img":
            cache.clear_img_cache()

        collection = [None] * n
        took = [0] * n
        if self.use_cache:
            # Assume the cache files are in order and complete
            num_graph_cache_files = min(len(cache.graph_cache_files), n)
            with ThreadPoolExecutor(max_workers=2 * async_workers) as thread_executor:
                thread_futures = {
                    thread_executor.submit(func, i, *args, **kwargs): i
                    for i in range(num_graph_cache_files)
                }
                if len(thread_futures) > 0:
                    print(f"Processing {len(thread_futures)} cache files...")
                for future in as_completed(thread_futures):
                    i = thread_futures[future]
                    collection[i], took[i] = future.result()
                    print(
                        f"Processed frame {i+1}/{n} from graph cache in {took[i]:.2f} seconds"
                    )
        else:
            num_graph_cache_files = 0
        if async_mode == "on":
            with ProcessPoolExecutor(max_workers=async_workers) as process_exector:
                # Process the remaining frames
                process_futures = {
                    process_exector.submit(func, i, *args, **kwargs): i
                    for i in range(num_graph_cache_files, n)
                }
                if len(process_futures) > 0:
                    print(f"Processing {len(process_futures)} frames asynchronously...")
                for future in as_completed(process_futures):
                    i = process_futures[future]
                    collection[i], took[i] = future.result()
                    print(f"Processed frame {i+1}/{n} in {took[i]:.2f} seconds")
        else:
            if not self.use_cache or n - num_graph_cache_files > 0:
                print(
                    f"Processing {abs(num_graph_cache_files - n)} frames synchronously..."
                )
            for i in range(num_graph_cache_files, n):
                collection[i], took[i] = func(i, *args, **kwargs)
                print(f"Processed frame {i+1}/{n} in {took[i]:.2f} seconds")
        avg_took = sum(took) / len(took)
        return collection, avg_took

    def generate_unique_filename(self, filename, hash_seed=None) -> str:
        """Generate a unique filename by appending a hash if the file already exists"""
        if os.path.exists(filename):
            hash = hashlib.sha1()
            if hash_seed is not None:
                hash.update(str(hash_seed).encode("utf-8"))
            else:
                hash.update(str(time.time()).encode("utf-8"))
            unique_hash = hash.hexdigest()[:5]
            basename, ext = os.path.splitext(filename)
            filename = f"{basename}_{unique_hash}{ext}"
        return filename

    def create_video(self, async_mode: str = "off", async_workers: int = 4) -> tuple:
        """Create a video from the images generated for each time frame and add the original audio."""
        # Gen_file and save an image for each time frame
        image_files, avg_took = self.process_time_frames(
            self.process_frame, async_mode, async_workers
        )

        # Get the size of and name image
        img = cv2.imread(image_files[0])
        height, width, layers = img.shape
        output_dir = f"output/{os.path.splitext(os.path.basename(self.filename))[0]}/"
        os.makedirs(output_dir, exist_ok=True)
        video_filename, output_filename = (
            f"{output_dir}video.mp4",
            f"{output_dir}output.mp4",
        )
        video_filename = self.generate_unique_filename(video_filename, self.filename)
        output_filename = self.generate_unique_filename(output_filename, self.filename)

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
            self.filename,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            output_filename,
            "-y",
        ]
        subprocess.run(command, check=True)
        return video_filename, output_filename, avg_took

    def render_frame(self, screen: pygame.Surface, graph: np.ndarray):
        """Render a single frame"""
        # Convert the graph to a pygame surface
        surface = pygame.surfarray.make_surface(graph)
        surface = pygame.transform.scale(surface, (self.size, self.size))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    def render(self, screen: pygame.Surface):
        # TODO: Implement this method
        pass
        # t = pygame.time.get_ticks()
        # getTicksLastFrame = t
        # # pygame.mixer.music.load(self.filename)
        # # pygame.mixer.music.play(0)

        # # Run until the user asks to quit
        # running = True
        # while running:
        #     # Did the user click the window close button?
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             running = False

        #     # Draw the graph
        #     self.process_frame(
        #         int(pygame.time.get_ticks() / 1000 * self.audio.time_index_ratio),
        #         screen,
        #     )

        #     # Cap the frame rate
        #     t = pygame.time.get_ticks()
        #     while t - getTicksLastFrame < 1000 / self.fps:
        #         t = pygame.time.get_ticks()
        #     getTicksLastFrame = t

        # Done! Time to quit.
        # pygame.quit()
