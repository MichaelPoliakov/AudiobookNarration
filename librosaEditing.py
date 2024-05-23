#!/usr/bin/env python 

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import json

class AudioProcessor:
    """
    A class to process audio files by removing silence and speeding up the audio.

    Attributes:
    - silence_thresh (float): Threshold below which audio is considered silence (in dB).
    - min_silence_duration (float): Minimum duration of silence to be removed (in seconds).
    - frame_length (int): Number of samples per frame for analysis.
    - hop_length (int): Number of samples between successive frames.
    - speed_factor (float): Factor by which to speed up the audio.
    - input_file (str): Path to the input audio file.
    - output_file (str): Path to the output audio file.
    """

    def __init__(self, config_file):
        """
        Initializes the AudioProcessor with parameters from a configuration file.
        Loads the configuration from a JSON file.

        Parameters:
        - config_file (str): Path to the configuration JSON file.
        """
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
                self.input_file = config.get("input_file", "input.aif")
                self.silence_thresh = config.get("silence_thresh", -40.0)
                self.min_silence_duration = config.get("min_silence_duration", 0.5)
                self.frame_length = config.get("frame_length", 4096)
                self.hop_length = config.get("hop_length", 1024)
                self.speed_factor = config.get("speed_factor", 1.5)
                self.output_file = config.get("output_file", "output.aif")
        except json.JSONDecodeError as e:
            print(f"Error loading configuration file: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def load_audio(self):
        """
        Loads the audio file.

        Returns:
        - y (numpy.ndarray): Audio time series.
        - sr (int): Sampling rate of the audio file.
        """
        try:
            y, sr = librosa.load(self.input_file, sr=None)
            return y, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None

    def detect_silence(self, y, sr):
        """
        Detects silent frames in the audio.

        Parameters:
        - y (numpy.ndarray): Audio time series.
        - sr (int): Sampling rate of the audio file.

        Returns:
        - silent_samples (numpy.ndarray): Indices of silent samples in the audio.
        """
        # Convert the silence threshold from dB to amplitude
        silence_thresh_amplitude = librosa.db_to_amplitude(self.silence_thresh)
        frame_length = self.frame_length or int(librosa.time_to_samples(0.023, sr))  # Default to 23 ms
        hop_length = self.hop_length or int(frame_length // 4)  # Default to 25% of frame length

        # Calculate the root mean square (RMS) energy for each frame
        rms_energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
        silent_frames = np.where(rms_energy < silence_thresh_amplitude)[0]
        silent_samples = librosa.frames_to_samples(silent_frames, hop_length=hop_length)

        # Debug: Print number of silent frames detected and the RMS energy values
        print(f"Silent frames detected: {len(silent_frames)}")
        print(f"RMS energy values: {rms_energy}")

        


        return silent_samples


    def get_non_silent_segments(self, silent_samples, y, sr):
        """
        Identifies non-silent segments in the audio.

        Parameters:
        - silent_samples (numpy.ndarray): Indices of silent samples in the audio.
        - y (numpy.ndarray): Audio time series.
        - sr (int): Sampling rate of the audio file.

        Returns:
        - non_silent_segments (list of tuples): List of tuples where each tuple represents the start and end indices of a non-silent segment.
        """
        non_silent_segments = []
        min_silence_duration_samples = int(self.min_silence_duration * sr)
        prev_end = 0

        for start in silent_samples:
            if start - prev_end > min_silence_duration_samples:
                non_silent_segments.append((prev_end, start))
            prev_end = start

        # Ensure the final segment is included if there is remaining non-silent audio
        if prev_end < len(y):
            non_silent_segments.append((prev_end, len(y)))

        # Merge consecutive non-silent segments if they are separated by less than min_silence_duration
        merged_segments = []
        for i in range(len(non_silent_segments)):
            if i == 0:
                merged_segments.append(non_silent_segments[i])
            else:
                prev_start, prev_end = merged_segments[-1]
                curr_start, curr_end = non_silent_segments[i]
                if curr_start - prev_end < min_silence_duration_samples:
                    merged_segments[-1] = (prev_start, curr_end)
                else:
                    merged_segments.append(non_silent_segments[i])
            

        # Debug: Print detailed segment information
        print(f"Detected non-silent segments (total: {len(merged_segments)}):")
        for segment in merged_segments:
            start_time = librosa.samples_to_time(segment[0], sr=sr)
            end_time = librosa.samples_to_time(segment[1], sr=sr)
            segment_length = segment[1] - segment[0]
            print(f"Segment from {start_time:.2f} sec to {end_time:.2f} sec, length: {segment_length} samples")

        return merged_segments





    def concatenate_non_silent_segments(self, non_silent_segments, y, sr):
        """
        Concatenates non-silent segments of the audio.

        Parameters:
        - non_silent_segments (list of tuples): List of tuples representing the start and end indices of non-silent segments.
        - y (numpy.ndarray): Audio time series.

        Returns:
        - non_silent_audio (numpy.ndarray): Concatenated non-silent audio.
        """
        non_silent_segments
        non_silent_audio = np.concatenate([y[start:end] for start, end in non_silent_segments])
        

        # Debug: Print the total length of the concatenated audio
        total_length = len(non_silent_audio)
        print(f"Total length of concatenated non-silent audio: {total_length} samples ({total_length / sr:.2f} seconds)")


        return non_silent_audio


    def speed_up_audio(self, y):
        """
        Speeds up the audio by the specified factor.

        Parameters:
        - y (numpy.ndarray): Audio time series.

        Returns:
        - sped_up_audio (numpy.ndarray): Sped-up audio time series.
        """
        try:
            if self.speed_factor == 1:
              return
            sped_up_audio = librosa.effects.time_stretch(y, rate=self.speed_factor)
            return sped_up_audio
        except Exception as e:
            print(f"Error speeding up audio: {e}")
            return None

    def save_audio(self, y, sr):
        """
        Saves the processed audio to the specified output file.

        Parameters:
        - y (numpy.ndarray): Audio time series.
        - sr (int): Sampling rate of the audio file.
        """
        try:
            # Specify the format explicitly based on the file extension
            file_extension = self.output_file.split('.')[-1].lower()
            if file_extension in ['wav', 'flac', 'ogg', 'aiff', 'aif']:
                sf.write(self.output_file, y, sr, format=file_extension.upper())
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            print(f"Processed file saved as: {self.output_file}")
        except Exception as e:
            print(f"Error saving audio file: {e}")


    def plot_waveforms(self, original_y, processed_y, sr):
        """
        Plots the waveforms of the original and processed audio.

        Parameters:
        - original_y (numpy.ndarray): Original audio time series.
        - processed_y (numpy.ndarray): Processed audio time series.
        - sr (int): Sampling rate of the audio files.
        """
        plt.figure(figsize=(14, 6))

        plt.subplot(2, 1, 1)
        plt.plot(original_y)
        plt.title('Original Audio')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        plt.plot(processed_y)
        plt.title('Processed Audio')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

    def process_audio(self):
        """
        Orchestrates the entire process of loading, processing, and saving the audio file.
        """
        print(f"Loading audio file: {self.input_file}")
        original_y, sr = self.load_audio()
        if original_y is None or sr is None:
            return

        print("Detecting silence...")
        silent_samples = self.detect_silence(original_y, sr)
        print(f"Silent samples detected: {len(silent_samples)}")

        print("Getting non-silent segments...")
        non_silent_segments = self.get_non_silent_segments(silent_samples, original_y, sr)
        print(f"Non-silent segments: {len(non_silent_segments)}")

        print("Concatenating non-silent segments...")
        non_silent_audio = self.concatenate_non_silent_segments(non_silent_segments, original_y, sr)
        print(f"Non-silent audio length: {len(non_silent_audio)} samples ({len(non_silent_audio) / sr:.2f} seconds)")
      
    def speed_up(self, factor=None):
      if factor is None:
        print("Not speeding up. Factor is x1.")
        return
      print("Speeding up audio...")
        sped_up_audio = self.speed_up_audio(non_silent_audio)
        if sped_up_audio is not None:
            print(f"Saving processed audio to: {self.output_file}")
            self.save_audio(sped_up_audio, sr)
            print("Plotting waveforms...")
            self.plot_waveforms(original_y, sped_up_audio, sr)
      

if __name__ == "__main__":
    config_file = "config.json"

    processor = AudioProcessor(config_file)
    processor.process_audio()
