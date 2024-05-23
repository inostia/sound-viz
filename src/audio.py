import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import cv2
import librosa
import numpy as np
from matplotlib import pyplot as plt

CACHE_DIR = "cache/"


class Audio:
    filename: str = ""
    time_series: np.ndarray = np.array([])
    sample_rate: int = 0
    stft: np.ndarray = np.array([])
    spectrogram: np.ndarray = np.array([])
    frequencies: np.ndarray = np.array([])
    times: np.ndarray = np.array([])
    time_index_ratio: float = 0.0
    frequencies_index_ratio: float = 0.0
    bpm: float = 0.0
    total_beats: int = 0
    cache_dir: str = ""

    def __init__(self, filename):
        self.filename = filename
        self.set_cache_dir()
        self.time_series, self.sample_rate = librosa.load(filename)
        self.stft = np.abs(librosa.stft(self.time_series, hop_length=512, n_fft=2048 * 4))
        self.spectrogram = librosa.amplitude_to_db(self.stft, ref=np.max)
        # getting an array of frequencies
        self.frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)
        # getting an array of time periodic
        self.times = librosa.core.frames_to_time(
            np.arange(self.spectrogram.shape[1]),
            sr=self.sample_rate,
            hop_length=512,
            n_fft=2048 * 4,
        )
        # self.time_index_ratio = len(self.times) / self.times[len(self.times) - 1]
        self.time_index_ratio = self.spectrogram.shape[1] / self.times[-1]
        self.frequencies_index_ratio = len(self.frequencies) / self.frequencies[len(self.frequencies) - 1]
        # self.bpm = self.detect_bpm()
        self.bpm = 145
        self.total_beats = self.get_total_beats()

    def set_cache_dir(self):
        """Set the grid cache directory"""
        self.cache_dir = f"{CACHE_DIR}{os.path.splitext(os.path.basename(self.filename))[0]}/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_grid_cache_item(self, i):
        """Return the grid cache item for a given index"""
        if not os.path.exists(f"{self.cache_dir}/grid/{i}.npy"):
            return
        grid = np.load(f"{self.cache_dir}/grid/{i}.npy")
        return grid

    def save_grid_cache_item(self, i, grid):
        """Save the grid cache to a file so it can be used later"""
        if not os.path.exists(f"{self.cache_dir}/grid"):
            os.makedirs(f"{self.cache_dir}/grid")
        np.save(f"{self.cache_dir}/grid/{i}.npy", grid)

    def display_spectrogram(self):
        """Display the spectrogram"""
        librosa.display.specshow(self.spectrogram, y_axis="log", x_axis="time")
        plt.title(self.filename)
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()

    def get_decibel(self, target_time, freq):
        """Get the decibel value at a given time and frequency"""
        return self.spectrogram[int(freq * self.frequencies_index_ratio)][int(target_time * self.time_index_ratio)]

    def get_spectrogram_slice(self, i):
        """Get the spectrogram slice for a given index in the time axis"""
        return self.spectrogram[:, i]

    def detect_bpm(self):
        """Detect the BPM of the audio file"""
        onset_env = librosa.onset.onset_strength(y=self.time_series, sr=self.sample_rate)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
        # Assume the BPM is constant throughout the audio file
        # TODO: handle the case where the BPM changes and find the BPM for each time frame
        return tempo[0]

    def get_total_beats(self):
        """Get the total number of beats in the audio file.
        Convert the total number of frames to the total number of beats using the BPM"""
        len_times = len(self.times)
        frames_per_beat = int(round(self.time_index_ratio * 60 / self.bpm))
        return len_times // frames_per_beat

    def process_time_frames(self, func, sync, *args, **kwargs):
        """Iterate over each time frame and pass the spectrogram slice to the given function"""
        n = len(self.times)
        collection = [None] * n  # Initialize a list of the same length as self.times
        if sync:
            print("Processing time frames...")
            for i in range(n):
                collection[i] = func(self, i, *args, **kwargs)
                print(f"Processed frame {i+1} of {n}")
        else:
            # with ThreadPoolExecutor(max_workers=60) as executor:
            with ProcessPoolExecutor(max_workers=12) as executor:
            # with ThreadPoolExecutor(max_workers=12) as executor:
                print("Processing time frames...")
                futures = {executor.submit(func, self, i, *args, **kwargs): i for i in range(n)}
                for future in as_completed(futures):
                    i = futures[future]
                    res = future.result()
                    collection[i] = res
                    print(f"Processed frame {i+1} of {n}")
        return collection

    def create_visualization(self, func, sync=True, *args, **kwargs):
        """Create a video from the images generated for each time frame and add the original audio."""
        basename = os.path.basename(self.filename)
        video_filename, output_filename = (
            f"output/{basename}_video.mp4",
            f"output/{basename}_output.mp4",
        )
        fps = self.time_index_ratio

        # Gen_file and save an image for each time frame
        image_files = self.process_time_frames(func, sync, *args, **kwargs)

        # Get the size of tage_file image
        img = cv2.imread(image_files[0])
        height, width, layers = img.shape

        # Create a VideoWriter object
        video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

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
