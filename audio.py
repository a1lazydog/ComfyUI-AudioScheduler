
import numpy as np
from scipy.fft import fft

class AudioData:
    def __init__(self, audio_file) -> None:
        
        # Extract the sample rate
        sample_rate = audio_file.frame_rate

        # Get the number of audio channels
        num_channels = audio_file.channels

        # Extract the audio data as a NumPy array
        audio_data = np.array(audio_file.get_array_of_samples())
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
    
    def get_channel_audio_data(self, channel: int):
        if channel < 0 or channel >= self.num_channels:
            raise IndexError(f"Channel '{channel}' out of range. total channels is '{self.num_channels}'.")
        return self.audio_data[channel::self.num_channels]
    
    def get_channel_fft(self, channel: int):
        audio_data = self.get_channel_audio_data(channel)
        return fft(audio_data)

class AudioFFTData:
    def __init__(self, audio_data, sample_rate) -> None:

        self.fft = fft(audio_data)
        self.length = len(self.fft)
        self.frequency_bins = np.fft.fftfreq(self.length, 1 / sample_rate)
    
    def get_max_amplitude(self):
        return np.max(np.abs(self.fft))
    
    def get_normalized_fft(self) -> float:
        max_amplitude = self.get_max_amplitude()
        return np.abs(self.fft) / max_amplitude

    def get_indices_for_frequency_bands(self, lower_band_range: int, upper_band_range: int):
        return np.where((self.frequency_bins >= lower_band_range) & (self.frequency_bins < upper_band_range))

    def __len__(self):
        return self.length
