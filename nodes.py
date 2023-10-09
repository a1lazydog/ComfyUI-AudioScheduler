import os
import folder_paths
import numpy as np
import pandas as pd

from pydub import AudioSegment
from scipy.fft import fft

from.audio import AudioData, AudioFFTData

from comfy.k_diffusion.utils import FolderOfImages

folder_paths.folder_names_and_paths["video_formats"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats"),
    ],
    [".json"]
)

class LoadAudioFromPath:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "file": ("STRING", {"default": ""}),
                     },}

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "load_audio"


    def load_audio(self, file: str):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File '{file} cannot be found.'")

        # TODO: support more formats
        mp3_file = AudioSegment.from_mp3(file)
        audio_data = AudioData(mp3_file)

        return (audio_data,)
     

class AudioToFFTs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": ("AUDIO",),
                    "frames_per_second": ("INT", {"default": 12, "min": 0, "max": 240, "step": 1}),
                     },}

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("AUDIO_FFT","INT",)
    RETURN_NAMES = ("AUDIO_FFT","total_frames")
    FUNCTION = "fft"

    def fft(self, audio, frames_per_second, ):
        audio_fft = audio.get_channel_fft(0)

        # Number of samples in the audio data
        num_samples = len(audio_fft)
        
        samples_per_frame = int(np.ceil(audio.sample_rate / frames_per_second))
        
        # Calculate the number of frames
        total_frames = int(np.ceil(num_samples / samples_per_frame))

        ffts = []
        
        for i in range(0, total_frames):
            # Extract the current frame of audio data
            frame = audio_fft[i * samples_per_frame : ((i + 1) * samples_per_frame)]
            
            ffts.append(AudioFFTData(frame, audio.sample_rate))

        return (ffts,total_frames,)
    

class BatchAmplitudeSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio_fft": ("AUDIO_FFT",),
                    "invert_normalized": ("BOOLEAN", {"default": False},),
                    "lower_band_range": ("INT", {"default": 500.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                    "upper_band_range": ("INT", {"default": 4000.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                     },}

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("FLOAT", "INT",)
    RETURN_NAMES = ("normalized_amplitude", "amplitude")
    FUNCTION = "animate"

    def animate(self, audio_fft, invert_normalized:bool, lower_band_range: int, upper_band_range: int,):
        max_frames = len(audio_fft)
        normalized_key_frame_series = pd.Series([np.nan for a in range(max_frames)])
        key_frame_series = pd.Series([np.nan for a in range(max_frames)])
        
        for i in range(0, max_frames):

            fft = audio_fft[i]
            
            # Normalize the FFT result to the range [0, 1]
            normalized_fft_result = fft.get_normalized_fft()

            indices = fft.get_indices_for_frequency_bands(lower_band_range, upper_band_range)
            normalized_key_frame = np.max(normalized_fft_result[indices])
            if (invert_normalized):
                normalized_key_frame = 1.0 - normalized_key_frame
            normalized_key_frame_series[i] = normalized_key_frame
            key_frame_series[i] = int(np.max(fft.fft[indices]))

        return (normalized_key_frame_series, key_frame_series,)
    

class AmplitudeSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio_fft": ("AUDIO_FFT",),
                    "invert_normalized": ("BOOLEAN", {"default": False},),
                    "lower_band_range": ("INT", {"default": 500.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                    "upper_band_range": ("INT", {"default": 4000.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                    "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                     },}

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("FLOAT", "INT",)
    RETURN_NAMES = ("normalized_amplitude", "amplitude")
    FUNCTION = "animate"

    def animate(self, audio_fft, invert_normalized:bool, lower_band_range: int, upper_band_range: int, current_frame: int,):

        fft = audio_fft[current_frame]
        
        # Normalize the FFT result to the range [0, 1]
        normalized_fft_result = fft.get_normalized_fft()

        indices = fft.get_indices_for_frequency_bands(lower_band_range, upper_band_range)
        normalized_key_frame = np.max(normalized_fft_result[indices])
        if (invert_normalized):
            normalized_key_frame = 1.0 - normalized_key_frame

        return (normalized_key_frame, int(np.max(fft.fft[indices])),)
    

    

NODE_CLASS_MAPPINGS = {
    "LoadAudioFromPath": LoadAudioFromPath,
    "AudioToFFTs": AudioToFFTs,
    "BatchAmplitudeSchedule": BatchAmplitudeSchedule,
    "AmplitudeSchedule": AmplitudeSchedule,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudioFromPath": "Load Audio From Path",
    "AudioToFFTs": "Audio to FFTs",
    "BatchAmplitudeSchedule": "Batch Amplitude Schedule",
    "AmplitudeSchedule": "Amplitude Schedule",
}