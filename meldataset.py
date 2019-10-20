import random
import torch
import torch.utils.data
import numpy as np
import librosa
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return np.exp(x) / C


def spectral_normalize(magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output


def spectral_de_normalize(magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    assert(np.min(y.data) >= -1)
    assert(np.max(y.data) <= 1)

    y = np.pad(y, (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2))
               , 'constant', constant_values=(0, 0))
    spec = librosa.feature.melspectrogram(y, hop_length=hop_size, win_length=win_size, center=center,
                                          sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)

    magnitudes = np.abs(spec)
    mel_output = spectral_normalize(magnitudes)
    return mel_output


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.sampling_rate = sampling_rate
        self.fmin = fmin
        self.fmax = fmax

    def __getitem__(self, index):
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav(filename)
        audio = audio / MAX_WAV_VALUE
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        if self.split:
            if len(audio) >= self.segment_size:
                max_audio_start = len(audio) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[audio_start:audio_start+self.segment_size]
            else:
                audio = np.pad(audio, (0, self.segment_size - len(audio)), 'constant')

        mel = mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size,
                              self.fmin, self.fmax, center=False)

        return (torch.FloatTensor(mel), torch.FloatTensor(audio), filename)

    def __len__(self):
        return len(self.audio_files)
