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
                    },                            
               "optional": {
                    "start_at_frame": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1}),
                    "limit_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                    }
                }

    CATEGORY = "AudioScheduler"

    RETURN_TYPES = ("AUDIO_FFT","INT",)
    RETURN_NAMES = ("AUDIO_FFT","total_frames")
    FUNCTION = "fft"

    def fft(self, audio, channel:int, frames_per_second:int, start_at_frame:int=0, limit_frames:int=0):
        if (frames_per_second <= 0):
            raise ValueError(f"frames_per_second cannot be 0 or negative: {frames_per_second}")
        
        audio_data = audio.get_channel_audio_data(channel)

        # Number of samples in the audio data
        total_samples = len(audio_data)
        
        samples_per_frame = int(np.ceil(audio.sample_rate / frames_per_second))
        
        # Calculate the number of frames
        total_frames = int(np.ceil(total_samples / samples_per_frame))

        if (np.abs(start_at_frame) > total_frames):
            raise IndexError(f"Absolute value of start_at_frame '{start_at_frame}' cannot exceed the total_frames '{total_frames}'")

        # If value is negative, start from the back
        if (start_at_frame < 0):
            start_at_frame = total_frames + start_at_frame

        ffts = []

        if (limit_frames > 0 and start_at_frame + limit_frames < total_frames):
            end_at_frame = start_at_frame + limit_frames
        else:
            end_at_frame = total_frames
        
        for i in range(start_at_frame, end_at_frame):
            i_next = (i + 1) * samples_per_frame

            if i_next >= total_samples:
                i_next = total_samples

            # Extract the current frame of audio data
            frame = audio_data[i * samples_per_frame : i_next]
            
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

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "fft"

    def fft(self, audio, channel:int, lower_band_range:int, upper_band_range:int):
        
        audio_fft = audio.get_channel_fft(channel)

        # Number of samples in the audio data
        num_samples = len(audio_fft)

        amplitudes = 2 / num_samples * np.abs(audio_fft) 
        
        # Calculate the frequency bins
        frequency_bins = np.fft.fftfreq(num_samples, 1/audio.sample_rate)
        indices = np.where((frequency_bins >= lower_band_range) & (frequency_bins < upper_band_range))

        plt.figure(figsize=(50, 6))
        plt.plot(frequency_bins[indices], amplitudes[indices])
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
                    "operation": (["avg","max","sum"], {"default": "max"}),
                    "lower_band_range": ("INT", {"default": 500.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                    "upper_band_range": ("INT", {"default": 4000.0, "min": 0.0, "max": 100000.0, "step": 1.0}),
                     },}

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("AMPLITUDE",)
    RETURN_NAMES = ("amplitude",)
    FUNCTION = "animate"

    def animate(self, audio_fft, operation, lower_band_range: int, upper_band_range: int,):
        if (lower_band_range > upper_band_range):
            raise ValueError(f"lower_band_range '{lower_band_range}' cannot be higher than upper_band_range '{upper_band_range}'")
        
        max_frames = len(audio_fft)
        key_frame_series = pd.Series([np.nan for a in range(max_frames)])
        
        for i in range(0, max_frames):

            fft = audio_fft[i]

            indices = fft.get_indices_for_frequency_bands(lower_band_range, upper_band_range)

            amplitude = (2 / len(fft)) * np.abs(fft.fft[indices])

            if "avg" in operation:
                key_frame_series[i] = np.mean(amplitude)
            elif "max" in operation:
                key_frame_series[i] = np.max(amplitude)
            elif "sum" in operation:
                key_frame_series[i] = np.sum(amplitude)

        return (key_frame_series,)

class NormalizeAmplitude:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "amplitude": ("AMPLITUDE",),
                    },                          
               "optional": {
                    "invert_normalized": ("BOOLEAN", {"default": False},),
                    }
                }

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("NORMALIZED_AMPLITUDE",)
    RETURN_NAMES = ("normalized_amp",)
    FUNCTION = "normalize"

    def normalize(self, amplitude, invert_normalized:bool=False,):
        normalized_amplitude = amplitude / np.max(amplitude)
        if (invert_normalized):
            normalized_amplitude = 1.0 - normalized_amplitude
        return (normalized_amplitude,)

class GateNormalizedAmplitude:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "normalized_amp": ("NORMALIZED_AMPLITUDE",),
                    "gate_normalized": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                }

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("NORMALIZED_AMPLITUDE",)
    RETURN_NAMES = ("normalized_amp",)
    FUNCTION = "gate"

    def gate(self, normalized_amp, gate_normalized:float,):
        gated_amp = np.where(normalized_amp > gate_normalized, normalized_amp, 0.0)
        return (gated_amp,)

class NormalizedAmplitudeToNumber:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "normalized_amp": ("NORMALIZED_AMPLITUDE",),
                     },}

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "convert"

    def convert(self, normalized_amp,):
        return (normalized_amp, normalized_amp.astype(int))


class NormalizedAmplitudeToGraph:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "normalized_amp": ("NORMALIZED_AMPLITUDE", {"forceInput": True}),
                    },}

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "graph"

    def graph(self, normalized_amp,):

        width = int(len(normalized_amp) / 10)
        if (width < 10):
            width = 10
        if (width > 100):
            width = 100
        
        plt.figure(figsize=(width, 6))
        plt.plot(normalized_amp,)

        plt.xlabel("Frame(s)")
        plt.ylabel("Amplitude")
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
    

class AmplitudeToNumber:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "amplitude": ("AMPLITUDE",),
                     },}

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "convert"

    def convert(self, amplitude,):
        return (amplitude.astype(float), amplitude.astype(int))


class AmplitudeToGraph:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "amplitude": ("AMPLITUDE", {"forceInput": True}),
                    },}

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "graph"

    def graph(self, amplitude,):

        width = int(len(amplitude) / 10)
        if (width < 10):
            width = 10
        if (width > 100):
            width = 100
        
        plt.figure(figsize=(width, 6))
        plt.plot(amplitude,)

        # Prevent scientific notation on the y-axis
        plt.ticklabel_format(axis='y', style='plain')

        plt.xlabel("Frame(s)")
        plt.ylabel("Amplitude")
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
    "NormalizeAmplitude": NormalizeAmplitude,
    "GateNormalizedAmplitude": GateNormalizedAmplitude,
    "NormalizedAmplitudeToNumber" : NormalizedAmplitudeToNumber,
    "NormalizedAmplitudeToGraph" : NormalizedAmplitudeToGraph,
    "AmplitudeToNumber" : AmplitudeToNumber,
    "AmplitudeToGraph" : AmplitudeToGraph,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudio": "Load Audio",
    "AudioToFFTs": "Audio to FFTs",
    "AudioToAmplitudeGraph": "Audio to Amplitude Graph",
    "BatchAmplitudeSchedule": "Batch Amplitude Schedule",
    "NormalizeAmplitude": "Normalize Amplitude",
    "GateNormalizedAmplitude": "Gate Normalized Amplitude",
    "NormalizedAmplitudeToNumber" : "Normalized Amplitude To Float or Int",
    "NormalizedAmplitudeToGraph" : "Normalized Amplitude To Graph",
    "AmplitudeToNumber" : "Amplitude To Float or Int",
    "AmplitudeToGraph" : "Amplitude To Graph",
}