import matplotlib.pyplot as plt
import numpy as np
import struct
import os
import json
import scipy.io.wavfile as wavfile

from enum import Enum
from docopt import docopt
from progress.bar import Bar
from moviepy.editor import AudioFileClip
from pydub import AudioSegment

SPECIES_RAW_AUDIO_PATH = "./../../data/audio/raw/"
SPECIES_PROCESSED_AUDIO_PATH = "./../../data/audio/processed/"
SPECTROGRAM_PATH = "./../../data/images/spectrograms/"
SPECIES_TO_CONVERT = []


class AudioFormat(Enum):
    WAV = 0
    MP3 = 1


def _dir_size(dir: str):
    ''' Returns no. of files for a given directory of species '''
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
    ''' Reads an MP3 file '''
    if format == AudioFormat.MP3:
        try:
            file = AudioSegment.from_mp3(f'{file_path}.mp3')
            return file
        except Exception as ex:
            print(f'Error reading: {file_path}. Reason="{ex}"')
            return
    else:
        # TODO: Implement other formats
        return


def split_mp3(file_path: str, interval: int = 10):
    start_time = 0
    end_time = interval
    try:
        audio = AudioSegment.from_mp3(f'{file_path}.mp3')
        duration = audio.duration_seconds
        if duration < interval + 5:
            return
        left_over = duration % interval
        num_splits = int(duration / interval)
        if left_over < 5:
            num_splits -= 1
        for i in range(0, num_splits):
            start_time = 10 * i * 1000  # milliseconds from a 10s interval
            end_time = (interval + (10 * i)) * 1000
            extract = audio[start_time:end_time]
            extract.export(f'{file_path}_{i}.mp3')
        extract = audio[10 * num_splits * 1000 : duration * 1000]
        extract.export(f'{file_path}_{num_splits}.mp3')
        os.system(f'rm {file_path}.mp3')
    except Exception as ex:
        print(f'ERROR: Failed to read {file_path}. Reason="{ex}"')


def process_split(interval: int = 10):
    print(f'Starting mp3 split using interval {interval}')
    for species in SPECIES_TO_CONVERT:
        num_files = _dir_size(f'{SPECIES_RAW_AUDIO_PATH}{species}/mp3')
        with Bar(
            f'Splitting MP3s for {species} in {interval}s intervals',
            suffix='%(percent)d%%',
            max=num_files,
        ) as bar:
            for i in range(1, num_files + 1):
                try:
                    split_mp3(
                        f'{SPECIES_RAW_AUDIO_PATH}{species}/mp3/{species}_{i}'
                    )
                except Exception as ex:
                    print(
                        f'ERROR: Failed to split {species}_{i}.mp3. Reason="{ex}"'
                    )
                bar.next()


def convert_wav(filename: str, species: str):
    ''' Converts an audio file into a mono channel WAV '''
    try:
        file = read_file(
            f'{SPECIES_RAW_AUDIO_PATH}{species}/mp3/{filename}',
            AudioFormat.MP3,
        )
        file.export(
            f'{SPECIES_PROCESSED_AUDIO_PATH}{species}/{filename}.wav',
            format="wav",
        )
        normalise_wav(
            f'{SPECIES_PROCESSED_AUDIO_PATH}{species}/{filename}.wav'
        )
        delete_mp3(filename, species)
    except Exception as ex:
        print(f'ERROR: Failed to export wav file {filename}. Reason="{ex}"')


def normalise_wav(filename: str):
    ''' Converts the stereo to mono '''
    rate, audio_data = wavfile.read(filename)
    if len(audio_data.shape) > 1:
        left_channel = audio_data[:, 0]
    else:
        left_channel = audio_data
    wavfile.write(f'{filename}', rate, left_channel)


def normalise_wavs():
    ''' Normalise all the WAV files for a species '''
    for species in SPECIES_TO_CONVERT:
        json_file = open(
            f'{SPECIES_RAW_AUDIO_PATH}{species}/json/{species}_1.json'
        )
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
    ''' Delete an mp3 file '''
    os.system(f'rm {SPECIES_RAW_AUDIO_PATH}{species}/mp3/{filename}.mp3')


def read_wav(filename: str, species: str):
    ''' Read processed file and return the audio data to be plotted. '''
    _, signal_data = wavfile.read(
        f'{SPECIES_PROCESSED_AUDIO_PATH}{species}/{filename}.wav'
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

    return single_channel_data


def process_to_wav():
    ''' Converts to WAV for all given species '''
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


def spectrogram(
    filename: str,
    species: str,
    nfft: int = 512,
    window=np.hamming(512),
    format=AudioFormat.MP3,
    frame_rate: int = 22050,
):
    ''' Plot a spectrogram for WAV file and save '''
    # TODO: Overlap?
    # TODO: Ignoring frequency range

    try:
        if format == AudioFormat.WAV:
            audio_data = read_wav(filename, species)
        elif format == AudioFormat.MP3:
            audio = AudioFileClip(
                f'{SPECIES_RAW_AUDIO_PATH}{species}/mp3/{filename}.mp3'
            )
            audio_data = audio.to_soundarray()
            audio_data = audio_data[:, 0]
        else:
            return NotImplementedError()
        fig = plt.figure()
        plt.specgram(
            audio_data, Fs=frame_rate, NFFT=nfft, window=window, cmap='inferno'
        )
        fig.savefig(f'{SPECTROGRAM_PATH}{species}/{filename}.jpg')
        plt.close(fig)
    except Exception as ex:
        print(
            f'ERROR: Failed to convert {filename} to Spectrogram. Reason="{ex}"'
        )


def generate_spectrograms(
    format=AudioFormat.MP3,
    nfft: int = 512,
    window=np.hamming(512),
    frame_rate: int = 22050,
):
    ''' Generates spectrograms for all required species '''
    for species in SPECIES_TO_CONVERT:
        _make_dir(f'{SPECTROGRAM_PATH}{species}')
        if format == AudioFormat.MP3:
            file_count = _dir_size(f'{SPECIES_RAW_AUDIO_PATH}{species}/mp3')
        elif format == AudioFormat.WAV:
            file_count = _dir_size(f'{SPECIES_PROCESSED_AUDIO_PATH}{species}')
        with Bar(
            f'Converting {species} to Spectrograms',
            suffix='%(percent)d%%',
            max=file_count,
        ) as bar:
            for file in os.listdir(f'{SPECIES_RAW_AUDIO_PATH}{species}/mp3'):
                if file.endswith('.mp3'):
                    if not os.path.isfile(
                        f'{SPECTROGRAM_PATH}{species}/{file[:-4]}.jpg'
                    ):
                        spectrogram(
                            f'{file[:-4]}',
                            species,
                            nfft,
                            window,
                            format,
                            frame_rate,
                        )
                bar.next()


def main():
    args = docopt(
        """
    Usage:
        audio_helper.py [options] <filename> <convert>
    
        <convert>: "wav", "spec-mp3", "spec-wav", "wave", "norm", "split"
    Options:
        --species NUM       No. of Species
        --all-species       For all species in input file
        --nfft NUM          NFFT for generating spectrogram
        --frame-rate NUM    Frame rate to use for spectrogram
        --split-interval    Interval size for splitting audio
    """
    )

    if args['<convert>'] not in (
        'wav',
        'spec-wav',
        'spec-mp3',
        'wave',
        'norm',
        'split',
    ):
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
    elif args['<convert>'] == 'spec-mp3':
        generate_spectrograms(format=AudioFormat.MP3)
    elif args['<convert>'] == 'spec-wav':
        generate_spectrograms(format=AudioFormat.WAV)
    elif args['<convert>'] == 'norm':
        normalise_wavs()
    elif args['<convert>'] == 'split':
        split_interval = 10
        if args['--split-interval']:
            split_interval = int(args['--split-interval'])
        process_split(split_interval)
    else:
        pass
        # TODO: Generate Wave


if __name__ == "__main__":
    main()
