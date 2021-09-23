"""
Raw Audio -> Mel-Spectrogram 변환기
"""
import os
from tqdm import tqdm
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState

AUDIO_DIR = "../wavs"
SPECT_DIR = "../spect"


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)


mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


dir_name, subdir_list, _ = next(os.walk(AUDIO_DIR))
print("Audio Directory:", dir_name)

# p111, p112, p113, ...
for speaker in tqdm(sorted(subdir_list)):
	if not os.path.exists(os.path.join(SPECT_DIR, speaker)):
		os.makedirs(os.path.join(SPECT_DIR, speaker))

	_, _, file_list = next(os.walk(os.path.join(dir_name, speaker)))
	prng = RandomState(int(speaker[1:]))


	# p111/1.wav, p111/2.wav, p111/3.wav, ... 
	for file in sorted(file_list):
		# wav = (M,) 1channel 16000 rate wav 기준
		wav, sample_rate = sf.read(os.path.join(dir_name, speaker, file))
		# remove drifting noise
		wav = signal.filtfilt(b, a, wav)
		# compute spectrogram
		wav = pySTFT(wav).T
		wav = np.dot(wav, mel_basis)
		wav = 20 * np.log10(np.maximum(min_level, wav)) - 16
		# spect = (N, n_mels)
		spect = np.clip((wav + 100) / 100, 0, 1)

		# save spectrogram
		np.save(os.path.join(SPECT_DIR, speaker, file[:-4]), spect.astype(np.float32))




















