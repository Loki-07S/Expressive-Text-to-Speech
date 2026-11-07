"""
Evaluation Script

Calculates MCD (Mel-Cepstral Distortion) and Pitch RMSE
for evaluation.

Save outputs for subjective MOS evaluation.
"""
import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class Evaluator:
    def __init__(self, sr=22050):
        self.sr = sr
        
    def extract_melspectrogram(self, y):
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=80, fmax=8000)
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def extract_pitch(self, y):
        """
        Extract f0 (pitch) using PyIN.
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=self.sr)
        # Replace NaNs with 0
        f0 = np.nan_to_num(f0)
        return f0
        
    def calculate_mcd(self, reference_audio, generated_audio):
        """
        Calculates Mel-Cepstral Distortion between reference and generated audio.
        Requires reference audio and generated audio to be paths.
        Uses fastdtw for alignment.
        """
        y_ref, _ = librosa.load(reference_audio, sr=self.sr)
        y_gen, _ = librosa.load(generated_audio, sr=self.sr)
        
        # 1. MFCC extraction for MCD
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=self.sr, n_mfcc=24) # Typically use 24-36 MFCCs
        mfcc_gen = librosa.feature.mfcc(y=y_gen, sr=self.sr, n_mfcc=24)
        
        # Exclude dynamic components (only use the static features)
        mfcc_ref = mfcc_ref[1:, :].T # Typically omit the 0th coefficient (energy)
        mfcc_gen = mfcc_gen[1:, :].T
        
        # 2. Dynamic Time Warping due to duration mismatch
        distance, path = fastdtw(mfcc_ref, mfcc_gen, dist=euclidean)
        
        # 3. MCD calculation based on aligned frames
        mcd = 0.0
        n_frames = len(path)
        for (i, j) in path:
            diff = mfcc_ref[i] - mfcc_gen[j]
            mcd += np.sqrt(np.sum(diff**2))
            
        # Standard MCD scaling (convert distance to MCD usually via constant 10/ln(10) * sqrt(2))
        const = (10.0 / np.log(10.0)) * np.sqrt(2.0)
        mcd = (const / n_frames) * mcd
        
        return mcd

    def calculate_pitch_rmse(self, reference_audio, generated_audio):
        """
        Calculates Pitch RMSE after dynamic time warping alignment.
        """
        y_ref, _ = librosa.load(reference_audio, sr=self.sr)
        y_gen, _ = librosa.load(generated_audio, sr=self.sr)
        
        f0_ref = self.extract_pitch(y_ref)
        f0_gen = self.extract_pitch(y_gen)
        
        # DTW on f0
        distance, path = fastdtw(f0_ref.reshape(-1, 1), f0_gen.reshape(-1, 1), dist=euclidean)
        
        rmse = 0.0
        n_frames = len(path)
        
        for (i, j) in path:
            diff = f0_ref[i] - f0_gen[j]
            rmse += (diff**2)
            
        rmse = np.sqrt(rmse / n_frames)
        
        return rmse

if __name__ == "__main__":
    import os
    print("Testing objective evaluation logic...")
    # NOTE: You must provide a reference audio and a generated audio to test.
    # evaluator = Evaluator()
    # print("MCD Score:", evaluator.calculate_mcd("ref.wav", "gen.wav"))
    # print("Pitch RMSE Score:", evaluator.calculate_pitch_rmse("ref.wav", "gen.wav"))
