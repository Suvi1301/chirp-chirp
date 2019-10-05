import pydub
import struct
import scipy.io.wavfile
from enum import Enum


raw_file_path = "./../data/raw/"
processed_file_path = "./../data/processed/"


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
