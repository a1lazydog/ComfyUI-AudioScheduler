import os
from io import BytesIO
import hashlib
import folder_paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random

from pydub import AudioSegment
from PIL import Image
from itertools import cycle

from.audio import AudioData, AudioFFTData

defaultPrompt="""Rabbit
Dog
Cat
One prompt per line
"""

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

class ClipAmplitude:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "amplitude": ("AMPLITUDE",),
                    "max_amplitude": ("INT", {"default": 1000, "min": 0, "step": 1}),
                    },
               "optional": {
                    "min_amplitude": ("INT", {"default": 0, "min": 0, "step": 1}),
                    }
                }

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("AMPLITUDE",)
    RETURN_NAMES = ("amplitude",)
    FUNCTION = "clip"

    def clip(self, amplitude, max_amplitude:int, min_amplitude:int=0):
        if (min_amplitude > max_amplitude):
            raise ValueError(f"min_amplitude '{min_amplitude}' cannot be higher than max_amplitude '{max_amplitude}'")

        clipped_amp = np.where(amplitude < max_amplitude, amplitude, max_amplitude)
        clipped_amp = np.where(min_amplitude < clipped_amp, clipped_amp, min_amplitude)
        return (clipped_amp,)

class TransientAmplitudeBasic:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "amplitude": ("AMPLITUDE",),
                    },
               "optional": {
                    "frames_to_attack": ("INT", {"default": 0, "min": 0, "step": 1}),
                    "frames_to_hold": ("INT", {"default": 6, "min": 0, "step": 1}),
                    "frames_to_release": ("INT", {"default": 6, "min": 0, "step": 1}),
                    }
                }

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("AMPLITUDE",)
    RETURN_NAMES = ("amplitude",)
    FUNCTION = "adjust"

    def adjust(self, amplitude, frames_to_attack:int, frames_to_hold:int, frames_to_release:int):
        if (frames_to_attack < 0):
            raise ValueError(f"frames_to_attack '{frames_to_attack}' cannot be negative")

        if (frames_to_hold < 0):
            raise ValueError(f"frames_to_hold '{frames_to_hold}' cannot be negative")

        if (frames_to_release < 0):
            raise ValueError(f"frames_to_release '{frames_to_release}' cannot be negative")

        if (len(amplitude) <= 1):
            return (amplitude,)

        if (frames_to_attack == 0 and frames_to_hold == 0 and frames_to_release == 0):
            return (amplitude,)

        # Calculate the rise factor based on the number of frames to attack
        rise_factor = 1 / (frames_to_attack + 1)

        # Calculate the decay factor based on the number of frames to release
        decay_factor = 1 / (frames_to_release + 1)
        
        holding_frame = 0
        local_max_amplitude = amplitude[0]
        prev_amplitude = amplitude[0]
        adjusted_amp = [prev_amplitude]
        for i in range(1, len(amplitude)):

            # attack
            if (amplitude[i] >= prev_amplitude):
                # reset and set new goal
                holding_frame = 0
                local_max_amplitude = amplitude[i]

                # rise to the goal
                if (frames_to_attack > 0):
                    prev_amplitude += local_max_amplitude * rise_factor
                    if (prev_amplitude > amplitude[i]):
                        prev_amplitude = amplitude[i]
                else:
                    prev_amplitude = amplitude[i]
                adjusted_amp.append(prev_amplitude)
                continue

            # hold
            if (frames_to_hold > 0 and holding_frame < frames_to_hold):
                holding_frame += 1
                adjusted_amp.append(prev_amplitude)
                continue

            # release
            if (frames_to_release > 0):
                prev_amplitude -= local_max_amplitude * decay_factor
                if (prev_amplitude < amplitude[i]):
                    prev_amplitude = amplitude[i]
                adjusted_amp.append(prev_amplitude)
                continue

            # no adjustments for this frame
            prev_amplitude = amplitude[i]
            adjusted_amp.append(prev_amplitude)

        return (adjusted_amp,)

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
    
class NormalizedAmplitudeDrivenString:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "normalized_amp": ("NORMALIZED_AMPLITUDE",),
                    "triggering_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },                          
               "optional": {
                    "loop": ("BOOLEAN", {"default": True},),
                    "shuffle": ("BOOLEAN", {"default": False},),
                    }
                }

    @classmethod
    def IS_CHANGED(self, text, normalized_amp, triggering_threshold, loop, shuffle):
        if shuffle:
            return float("nan")
        m = hashlib.sha256()
        m.update(text)
        m.update(normalized_amp)
        m.update(triggering_threshold)
        m.update(loop)
        return m.digest().hex()


    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "convert"
        

    def convert(self, text, normalized_amp, triggering_threshold, loop, shuffle):
        prompts = text.splitlines()

        keyframes = self.get_keyframes(normalized_amp, triggering_threshold)

        if loop and len(prompts) < len(keyframes): # Only loop if there's more prompts than keyframes
            i = 0
            result = []
            for _ in range(len(keyframes) // len(prompts)):
                if shuffle:
                    random.shuffle(prompts)
                for prompt in prompts:
                    result.append('"{}": "{}"'.format(keyframes[i], prompt))
                    i += 1
        else: # normal
            if shuffle:
                random.shuffle(prompts)
            result = ['"{}": "{}"'.format(keyframe, prompt) for keyframe, prompt in zip(keyframes, prompts)]

        result_string = ',\n'.join(result)

        return (result_string,)

    def get_keyframes(self, normalized_amp, triggering_threshold):
        above_threshold = normalized_amp >= triggering_threshold
        above_threshold = np.insert(above_threshold, 0, False)  # Add False to the beginning
        transition = np.diff(above_threshold.astype(int))
        keyframes = np.where(transition == 1)[0]
        return keyframes

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
    # Amplitude
    "BatchAmplitudeSchedule": BatchAmplitudeSchedule,
    "ClipAmplitude": ClipAmplitude,
    "TransientAmplitudeBasic": TransientAmplitudeBasic,
    "AmplitudeToNumber" : AmplitudeToNumber,
    "AmplitudeToGraph" : AmplitudeToGraph,
    # Normalized Amplitude
    "NormalizeAmplitude": NormalizeAmplitude,
    "GateNormalizedAmplitude": GateNormalizedAmplitude,
    "NormalizedAmplitudeToNumber" : NormalizedAmplitudeToNumber,
    "NormalizedAmplitudeToGraph" : NormalizedAmplitudeToGraph,
    "NormalizedAmplitudeDrivenString" : NormalizedAmplitudeDrivenString
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudio": "Load Audio",
    "AudioToFFTs": "Audio to FFTs",
    "AudioToAmplitudeGraph": "Audio to Amplitude Graph",
    # Amplitude
    "BatchAmplitudeSchedule": "Batch Amplitude Schedule",
    "ClipAmplitude": "Clip Amplitude",
    "TransientAmplitudeBasic": "Transient Amplitude Basic",
    "AmplitudeToNumber" : "Amplitude To Float or Int",
    "AmplitudeToGraph" : "Amplitude To Graph",
    # Normalized Amplitude
    "NormalizeAmplitude": "Normalize Amplitude",
    "GateNormalizedAmplitude": "Gate Normalized Amplitude",
    "NormalizedAmplitudeToNumber" : "Normalized Amplitude To Float or Int",
    "NormalizedAmplitudeToGraph" : "Normalized Amplitude To Graph",
    "NormalizedAmplitudeDrivenString" : "Normalized Amplitude Driven String"
}
