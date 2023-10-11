import os
from io import BytesIO
import hashlib
import folder_paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pydub import AudioSegment
from PIL import Image

from.audio import AudioData, AudioFFTData

from comfy.k_diffusion.utils import FolderOfImages

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        audio_extensions = ['mp3','wav']
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {"required": {
                    "audio": (sorted(files),),
                     },}

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "load_audio"


    def load_audio(self, audio):
        file = folder_paths.get_annotated_filepath(audio)

        # TODO: support more formats
        if (file.lower().endswith('.mp3')):
            audio_file = AudioSegment.from_mp3(file)
        else:
            audio_file = AudioSegment.from_file(file, format="wav")
        
        audio_data = AudioData(audio_file)

        return (audio_data,)
    
    @classmethod
    def IS_CHANGED(self, audio, **kwargs):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(self, audio, **kwargs):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)

        return True
     

class AudioToFFTs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": ("AUDIO",),
                    "channel": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1}),
                    "frames_per_second": ("INT", {"default": 12, "min": 0, "max": 240, "step": 1}),
                    "start_at_frame": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1}),
                     },}

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("AUDIO_FFT","INT",)
    RETURN_NAMES = ("AUDIO_FFT","total_frames")
    FUNCTION = "fft"

    def fft(self, audio, channel:int, frames_per_second:int, start_at_frame:int,):
        if (frames_per_second <= 0):
            raise ValueError(f"frames_per_second cannot be 0 or negative: {frames_per_second}")
        
        audio_fft = audio.get_channel_fft(channel)

        # Number of samples in the audio data
        num_samples = len(audio_fft)
        
        samples_per_frame = int(np.ceil(audio.sample_rate / frames_per_second))
        
        # Calculate the number of frames
        total_frames = int(np.ceil(num_samples / samples_per_frame))

        if (np.abs(start_at_frame) > total_frames):
            raise IndexError(f"Absolute value of start_at_frame '{start_at_frame}' cannot exceed the total_frames '{total_frames}'")

        # If value is negative, start from the back
        if (start_at_frame < 0):
            start_at_frame = total_frames + start_at_frame

        ffts = []
        
        for i in range(start_at_frame, total_frames):
            # Extract the current frame of audio data
            frame = audio_fft[i * samples_per_frame : ((i + 1) * samples_per_frame)]
            
            ffts.append(AudioFFTData(frame, audio.sample_rate))

        return (ffts,total_frames,)
    

class AudioToAmplitudeGraph:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": ("AUDIO",),
                    "channel": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1}),
                    "lower_band_range": ("INT", {"default": 500.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                    "upper_band_range": ("INT", {"default": 4000.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                     },}

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "fft"

    def fft(self, audio, channel:int, lower_band_range:int, upper_band_range:int):
        
        audio_fft = audio.get_channel_fft(channel)

        # Number of samples in the audio data
        num_samples = len(audio_fft)
        
        # Calculate the frequency bins
        frequency_bins = np.fft.fftfreq(num_samples, 1/audio.sample_rate)

        indices = np.where((frequency_bins >= lower_band_range) & (frequency_bins < upper_band_range))

        plt.figure(figsize=(10, 4))
        plt.plot(frequency_bins[indices], np.abs(audio_fft[indices]))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("FFT of Audio Signal")

         # Create an in-memory buffer to store the image
        buffer = BytesIO()

        # Save the plot to the in-memory buffer as a PNG
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Create a Pillow Image object
        image = Image.open(buffer)

        return (pil2tensor(image),)

class BatchAmplitudeSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio_fft": ("AUDIO_FFT",),
                    "invert_normalized": ("BOOLEAN", {"default": False},),
                    "gate_normalized": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "lower_band_range": ("INT", {"default": 500.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                    "upper_band_range": ("INT", {"default": 4000.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                     },}

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("FLOAT", "INT",)
    RETURN_NAMES = ("normalized_amplitude", "amplitude")
    FUNCTION = "animate"

    def animate(self, audio_fft, invert_normalized:bool, gate_normalized:float, lower_band_range: int, upper_band_range: int,):
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

            if (normalized_key_frame > gate_normalized):
                normalized_key_frame_series[i] = normalized_key_frame
            else:
                normalized_key_frame_series[i] = 0.0
            key_frame_series[i] = int(np.max(fft.fft[indices]))

        return (normalized_key_frame_series, key_frame_series,)
    

class AmplitudeSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio_fft": ("AUDIO_FFT",),
                    "invert_normalized": ("BOOLEAN", {"default": False},),
                    "gate_normalized": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "lower_band_range": ("INT", {"default": 500.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                    "upper_band_range": ("INT", {"default": 4000.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                    "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                     },}

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("FLOAT", "INT",)
    RETURN_NAMES = ("normalized_amplitude", "amplitude")
    FUNCTION = "animate"

    def animate(self, audio_fft, invert_normalized:bool, gate_normalized:float, lower_band_range: int, upper_band_range: int, current_frame: int,):

        fft = audio_fft[current_frame]
        
        # Normalize the FFT result to the range [0, 1]
        normalized_fft_result = fft.get_normalized_fft()

        indices = fft.get_indices_for_frequency_bands(lower_band_range, upper_band_range)
        normalized_key_frame = np.max(normalized_fft_result[indices])
        if (invert_normalized):
            normalized_key_frame = 1.0 - normalized_key_frame

        if (normalized_key_frame > gate_normalized):
            normalized_key_frame = normalized_key_frame
        else:
            normalized_key_frame = 0.0

        return (normalized_key_frame, int(np.max(fft.fft[indices])),)


class FloatsToGraph:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "floats": ("FLOAT", {"forceInput": True}),
                    },}

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "graph"

    def graph(self, floats,):
        if (isinstance(floats, float)):
            raise ValueError(f"floats must be a list of value")
        
        plt.figure(figsize=(50, 6))
        plt.plot(floats,)

        plt.xlabel("Frame(s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()

        # Create an in-memory buffer to store the image
        buffer = BytesIO()

        # Save the plot to the in-memory buffer as a PNG
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Create a Pillow Image object
        image = Image.open(buffer)

        return (pil2tensor(image),)
    

NODE_CLASS_MAPPINGS = {
    "LoadAudio": LoadAudio,
    "AudioToFFTs": AudioToFFTs,
    "AudioToAmplitudeGraph": AudioToAmplitudeGraph,
    "BatchAmplitudeSchedule": BatchAmplitudeSchedule,
    "AmplitudeSchedule": AmplitudeSchedule,
    "FloatsToGraph" : FloatsToGraph,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudio": "Load Audio",
    "AudioToFFTs": "Audio to FFTs",
    "AudioToAmplitudeGraph": "Audio to Amplitude Graph",
    "BatchAmplitudeSchedule": "Batch Amplitude Schedule",
    "AmplitudeSchedule": "Amplitude Schedule",
    "FloatsToGraph" : "Floats To Graph",
}