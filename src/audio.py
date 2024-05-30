import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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

    def __init__(self, filename):
        self.filename = filename
        self.time_series, self.sample_rate = librosa.load(filename)
        self.hop_length = 512
        self.stft = np.abs(librosa.stft(self.time_series, hop_length=self.hop_length, n_fft=2048 * 4))
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
        self.bpm = self.detect_bpm()
        # self.bpm = 145

    def display_spectrogram(self):
        """Display the spectrogram"""
        librosa.display.specshow(self.spectrogram, y_axis="log", x_axis="time")
        plt.title(self.filename)
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()

    def get_decibel(self, target_time: int, freq: int):
        """Get the decibel value at a given time and frequency"""
        return self.spectrogram[int(freq * self.frequencies_index_ratio)][target_time]
    
    def get_energy(self, target_time: int, min_freq: int, max_freq: int):
        """Get the energy within a given frequency range at a given time"""
        min_freq_index = int(min_freq * self.frequencies_index_ratio)
        max_freq_index = int(max_freq * self.frequencies_index_ratio)
        # return np.sum(self.spectrogram[min_freq_index:max_freq_index, target_time])
        # Convert the db values to linear scale
        return np.sum(10 ** (self.spectrogram[min_freq_index:max_freq_index, target_time] / 10))


    def get_spectrogram_slice(self, target_time: int):
        """Get the spectrogram slice for a given index in the time axis"""
        return self.spectrogram[:, target_time]

    def detect_bpm(self):
        """Detect the BPM of the audio file"""
        onset_env = librosa.onset.onset_strength(y=self.time_series, sr=self.sample_rate)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
        # Assume the BPM is constant throughout the audio file
        # TODO: handle the case where the BPM changes and find the BPM for each time frame
        return tempo[0]
