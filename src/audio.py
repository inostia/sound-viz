import librosa
import numpy as np
import scipy.ndimage.filters
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
    bpm: float | None = None
    time_signature: str = "4/4"
    total_beats: int = 0

    def __init__(self, filename, bpm=None, time_signature="4/4", fps=30):
        self.filename = filename
        self.time_series, self.sample_rate = librosa.load(filename)
        self.hop_length = int(self.sample_rate / fps)
        self.stft = np.abs(
            librosa.stft(self.time_series, hop_length=self.hop_length, n_fft=2048 * 4)
        )
        self.spectrogram = librosa.amplitude_to_db(self.stft, ref=np.max)
        # getting an array of frequencies
        self.frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)
        # getting an array of time periodic
        self.times = librosa.core.frames_to_time(
            np.arange(self.spectrogram.shape[1]),
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=2048 * 4,
        )
        # self.time_index_ratio = len(self.times) / self.times[len(self.times) - 1]
        self.time_index_ratio = self.spectrogram.shape[1] / self.times[-1]
        self.frequencies_index_ratio = (
            len(self.frequencies) / self.frequencies[len(self.frequencies) - 1]
        )
        if bpm is not None:
            self.bpm = bpm
        else:
            self.bpm = self.detect_bpm()
        self.time_signature = time_signature

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

    def get_energy(
        self,
        target_time: int,
        min_freq: int,
        max_freq: int,
        freq_scale_factor: float | None = None,
        freq_scale_weight: int = 100,
    ):
        """Get the energy within a given frequency range at a given time

        Args:
            target_time (int): The time index
            min_freq (int): The minimum frequency
            max_freq (int): The maximum frequency
            freq_scale_factor (float): The frequency scale factor. A float between 0 and 1 will increase the amplitudes of the frequencies either toward the low end or high end of the spectrogram. Defaults to None.
            freq_scale_weight (int): The frequency scale weight. The weight of the Gaussian filter applied to the scaled spectrogram. Defaults to 100."""
        min_freq_index = int(min_freq * self.frequencies_index_ratio)
        max_freq_index = int(max_freq * self.frequencies_index_ratio)
        # return np.sum(self.spectrogram[min_freq_index:max_freq_index, target_time])
        # Convert the db values to linear scale
        # return np.sum(10 ** (self.spectrogram[min_freq_index:max_freq_index, target_time] / 10))
        spectrogram_linear = 10 ** (self.spectrogram / 10)
        if freq_scale_factor is not None:
            # Weight is a float between 0 and 1. weight either the low or high frequencies in the spectrogram
            # with a Gaussian filter
            freq_scale_factor = np.clip(freq_scale_factor, 0.0, 1.0)
            # Increase the amplitudes of the frequencies either toward the
            # the low end or high end of the spectrogram depending on the weight
            spectrogram_linear = scipy.ndimage.filters.gaussian_filter1d(
                # Always ensure the sigma is greater than 0
                spectrogram_linear,
                sigma=freq_scale_factor * freq_scale_weight + 0.001,
                axis=0,
            )
        return np.sum(spectrogram_linear[min_freq_index:max_freq_index, target_time])

    def get_spectrogram_slice(self, target_time: int):
        """Get the spectrogram slice for a given index in the time axis"""
        return self.spectrogram[:, target_time]

    def detect_bpm(self):
        """Detect the BPM of the audio file"""
        onset_env = librosa.onset.onset_strength(
            y=self.time_series, sr=self.sample_rate
        )
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=self.sample_rate
        )
        # Assume the BPM is constant throughout the audio file
        # TODO: handle the case where the BPM changes and find the BPM for each time frame
        return tempo[0]

    def total_beats_in_measures(self, measures: int) -> float:
        """Get the number of beats in n measures.
        
        Args:
            measures (int): The number of measures"""
        beats, unit = self.parse_time_signature()
        return measures * beats * (4 / unit)

    def parse_time_signature(self):
        """Parse the time signature"""
        if not self.time_signature:
            raise ValueError("Time signature is not set")

        time_signature = self.time_signature.split("/")
        if len(time_signature) != 2:
            raise ValueError(f"Invalid time signature: {self.time_signature}")

        try:
            beats = int(time_signature[0])
            unit = int(time_signature[1])
        except ValueError:
            raise ValueError(f"Time signature must be two integers separated by a slash: {self.time_signature}")

        if beats <= 0 or unit not in [1, 2, 4, 8, 16, 32, 64]:
            raise ValueError(f"Invalid time signature: {self.time_signature}")

        return beats, unit
    
    def get_beat(self, target_time: int, fps: int) -> float:
        """Get the beat at a given time"""
        t = target_time / fps
        return self.bpm * t / 60
