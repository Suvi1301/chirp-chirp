import matplotlib.pyplot as plt
import numpy as np
import pydub
import struct
import scipy.io.wavfile as wavfile
from enum import Enum


raw_file_path = "./../data/raw/"
processed_file_path = "./../data/processed/"
spectrograms_file_path = "./../data/spectrograms/"


class AudioFormat(Enum):
    WAV = 0
    MP3 = 1


def read_file(filename: str, format=AudioFormat.MP3):

    if format == AudioFormat.MP3:
        try:
            file = pydub.AudioSegment.from_mp3(
                f'{raw_file_path}{filename}.mp3'
            )
            return file
        except Exception as e:
            print(f'Error reading filename: {filename}. Reason="{e}"')
            return
    else:
        # TODO: Implement other formats
        return


def convert_wav(filename: str):
    file = read_file(filename, AudioFormat.MP3)
    file.export(f'{processed_file_path}{filename}.wav', format="wav")


def read_wav(filename: str):
    ''' Read processed file and return the audio data to be plotted. '''
    sampling_freq, signal_data = wavfile.read(
        f'{processed_file_path}{filename}.wav'
    )
    try:
        if len(signal_data.shape) > 1:
            single_channel_data = signal_data[:, 0]
        else:
            single_channel_data = signal_data
    except IndexError as ex:
        print(
            f'Error reading single channel from filename: {filename}. Reason="{ex}"'
        )

    return (sampling_freq, single_channel_data)


def generate_spectrogram(
    filename: str, nfft: int = 512, window=np.hamming(512)
):
    ''' Plot a spectrogram for audio file and save '''
    sampling_freq, signal_data = read_wav(filename)
    fig = plt.figure()
    # TODO: Figure out overlap.
    plt.specgram(signal_data, Fs=sampling_freq, NFFT=nfft, window=window)
    # TODO: Modify plot to ignore axis labels etc.
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    fig.savefig(f'{spectrograms_file_path}{filename}.png')
