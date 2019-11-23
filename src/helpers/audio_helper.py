import matplotlib.pyplot as plt
import numpy as np
import pydub
import struct
import os
import json
import scipy.io.wavfile as wavfile

from enum import Enum
from docopt import docopt
from progress.bar import Bar

SPECIES_RAW_AUDIO_PATH = "./../data/audio/raw/"
SPECIES_PROCESSED_AUDIO_PATH = "./../data/audio/processed/"
spectrograms_file_path = "./../data/images/"
SPECIES_TO_CONVERT = []


class AudioFormat(Enum):
    WAV = 0
    MP3 = 1


def _dir_size(dir: str):
    try:
        onlyfiles = next(os.walk(dir))[2]
        return len(onlyfiles)
    except OSError:
        print(f'ERROR: Could not read no. of files in {dir}')


def _make_dir(dir: str):
    ''' Create a new directory if not already exists '''
    try:
        os.mkdir(dir)
    except FileExistsError as ex:
        print(f'Directory ({dir}) already exists.')
    except OSError as ex:
        print(f'ERROR: Failed to create directory {dir}. Reason="{ex}"')


def read_file(file_path: str, format=AudioFormat.MP3):
    if format == AudioFormat.MP3:
        try:
            file = pydub.AudioSegment.from_mp3(f'{file_path}.mp3')
            return file
        except Exception as ex:
            print(f'Error reading: {file_path}. Reason="{ex}"')
            return
    else:
        # TODO: Implement other formats
        return


def convert_wav(filename: str, species: str):
    try:
        file = read_file(
            f'{SPECIES_RAW_AUDIO_PATH}{species}/mp3/{filename}',
            AudioFormat.MP3,
        )
        file.export(
            f'{SPECIES_PROCESSED_AUDIO_PATH}{species}/{filename}.wav',
            format="wav",
        )
        normalise_wav(f'{SPECIES_PROCESSED_AUDIO_PATH}{species}/{filename}.wav')
        delete_mp3(filename, species)
    except Exception as ex:
        print(f'ERROR: Failed to export wav file {filename}. Reason="{ex}"')


def normalise_wav(filename: str):
    rate, audio_data = wavfile.read(filename)
    if len(audio_data.shape) > 1:
        left_channel = audio_data[:, 0]
    else:
        left_channel = audio_data
    wavfile.write(f'{filename}', rate, left_channel)


def normalise_wavs():
    for species in SPECIES_TO_CONVERT:
        json_file = open(f'{SPECIES_RAW_AUDIO_PATH}{species}/json/{species}_1.json')
        data = json.load(json_file)
        json_file.close()
        num_files = int(data['numRecordings'])
        if num_files > 0:
            with Bar(
                f'Normalising {species} files to WAV',
                suffix='%(percent)d%%',
                max=num_files,
            ) as bar:
                for i in range(1, num_files + 1):
                    try:
                        normalise_wav(
                            f'{SPECIES_PROCESSED_AUDIO_PATH}{species}/{species}_{i}.wav'
                        )
                    except Exception as ex:
                        print(
                            f'ERROR: Failed to noramlise {species}_{i}.wav. Reason="{ex}"'
                        )
                    bar.next()


def delete_mp3(filename: str, species):
    os.system(f'rm {SPECIES_RAW_AUDIO_PATH}{species}/mp3/{filename}.mp3')


def read_wav(filename: str):
    ''' Read processed file and return the audio data to be plotted. '''
    sampling_freq, signal_data = wavfile.read(
        f'{SPECIES_PROCESSED_AUDIO_PATH}{filename}.wav'
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


def process_to_wav():
    for species in SPECIES_TO_CONVERT:
        _make_dir(f'{SPECIES_PROCESSED_AUDIO_PATH}{species}')
        file_count = _dir_size(f'{SPECIES_RAW_AUDIO_PATH}{species}/mp3')

        with Bar(
            f'Converting {species} files to WAV',
            suffix='%(percent)d%%',
            max=file_count,
        ) as bar:
            for i in range(1, file_count + 1):
                if not os.path.isfile(
                    f'{SPECIES_PROCESSED_AUDIO_PATH}{species}/{species}_{i}.wav'
                ):
                    convert_wav(f'{species}_{i}', species)
                bar.next()


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


def main():
    args = docopt(
        """
    Usage:
        audio_helper.py [options] <filename> <convert>
    
        <convert>: "wav", "spec", "wave", "norm"
    Options:
        --species NUM       No. of Species
        --all-species       For all species in input file
    """
    )

    if args['<convert>'] not in ('wav', 'spec', 'wave', 'norm'):
        print('Invalid argument for <convert>')
        return

    species_count = 1

    if args['--all-species']:
        species_count = 100

    if args['--species']:
        try:
            species_count = int(args['--species'])
        except TypeError:
            print(f'ERROR: --species must be an integer')
    else:
        print(f'Desired Species count not provided. Using {species_count}')

    with open(args['<filename>']) as input_file:
        line = input_file.readline()
        line_count = 0
        while line and line_count < species_count:
            SPECIES_TO_CONVERT.append(line.strip())
            line = input_file.readline()
            line_count += 1

    if args['<convert>'] == 'wav':
        process_to_wav()
    elif args['<convert>'] == 'spec':
        pass
        # TODO: Generate Spectrogram
    elif args['<convert>'] == 'norm':
        normalise_wavs()
    else:
        pass
        # TODO: Generate Wave


if __name__ == "__main__":
    main()
