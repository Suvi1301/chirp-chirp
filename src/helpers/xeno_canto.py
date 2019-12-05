import json
import urllib.request
import copy
import sys
import os
import wget
import pandas as pd

from docopt import docopt
from progress.bar import Bar

XENO_CANTO_URL = 'https://www.xeno-canto.org/api/2/recordings?query='
DESIRED_RECORDING_ATT = ('en', 'file', 'id', 'length', 'cnt')
SPECIES_TO_DOWNLOAD = []
SPECIES_PATH = './../../data/audio/raw/'


def _make_dir(dir: str):
    ''' Create a new directory if not already exists '''
    try:
        os.mkdir(dir)
    except FileExistsError as ex:
        print(f'Directory ({dir}) already exists.')
    except OSError as ex:
        print(f'ERROR: Failed to create directory {dir}. Reason="{ex}"')


def _dir_size(species: str, data: str = 'mp3'):
    ''' Returns no. of files for a given directory of species '''
    try:
        onlyfiles = next(os.walk(f'{SPECIES_PATH}{species}/{data}'))[2]
        return len(onlyfiles)
    except OSError:
        print(f'ERROR: Could not read no. of files in {species}')


def download_json_data():
    ''' Download JSON data from xeno-canto API into files for SPECIES_TO_DOWNLOAD '''
    for species in SPECIES_TO_DOWNLOAD:
        _make_dir(f'{SPECIES_PATH}{species}')
        _make_dir(f'{SPECIES_PATH}{species}/json')
        data = get_json(species)
        if _dir_size(species, 'json') < data['numPages']:
            save_json(species, data)
            print(f'Saved page 1 for {species}')
            if data['numPages'] > 1:
                i = 2
                with Bar(
                    f'Saving {data["numPages"]} pages for {species}',
                    suffix='%(percent)d%%',
                    max=data['numPages'] - 1,
                ) as bar:
                    while i <= data['numPages']:
                        data = get_json(species, page=i)
                        save_json(species, data)
                        i += 1
                        bar.next()
                    print(f'\n{data["numPages"]} pages saved for {species}')
        else:
            print(
                f'All JSON already exist for {species}. {_dir_size(species, "json")}/{data["numPages"]}'
            )


def download_audio_data():
    ''' Download the MP3 audio files for each SPECIES_TO_DOWNLOAD '''

    def read_json_file(filename: str):
        ''' Reads a given JSON file into dict '''
        json_file = open(filename)
        data = json.load(json_file)
        json_file.close()
        return data

    for species in SPECIES_TO_DOWNLOAD:
        _make_dir(f'{SPECIES_PATH}{species}/mp3')
        data_p1 = read_json_file(
            f'{SPECIES_PATH}{species}/json/{species}_1.json'
        )

        if _dir_size(species) < int(data_p1['numRecordings']):
            record_df = pd.DataFrame(data_p1['recordings'])

            for i in range(2, data_p1['numPages'] + 1):
                data = read_json_file(
                    f'{SPECIES_PATH}{species}/json/{species}_{i}.json'
                )
                record_df = record_df.append(pd.DataFrame(data['recordings']))

            record_df.to_csv(index=False)

            url_list = []
            for file in record_df['file'].tolist():
                url_list.append(f'https:{file}')

            with open(
                f'{SPECIES_PATH}{species}/{species}_urls.txt', 'w+'
            ) as f:
                i = 1
                print(f'Writing urls for {species} to file')
                with Bar(
                    f'Downloading MP3s for {species}',
                    suffix='%(percent)d%%',
                    max=len(url_list),
                ) as bar:
                    for url in url_list:
                        if not os.path.isfile(
                            f'{SPECIES_PATH}{species}/mp3/{species}_{i}.mp3'
                        ):
                            f.write(f'{url}\n')
                            wget.download(
                                url,
                                out=f'{SPECIES_PATH}{species}/mp3/{species}_{i}.mp3',
                            )
                        i += 1
                        bar.next()
        else:
            print(
                f'All mp3 already exist for {species}. {_dir_size(species)}/{data_p1["numRecordings"]}'
            )


def get_json(species: str, page: int = 1):
    ''' Request JSON from xeno-canto '''
    spec = species.split('_')
    query = spec[0]

    def relevant_json(data: json):
        ''' Removes unwanted fields in returned JSON from xeno-canto '''
        rel_json = copy.deepcopy(data)
        rel_json['recordings'] = []
        for recording in data['recordings']:
            rel_json['recordings'].append(
                {
                    k: v
                    for k, v in recording.items()
                    if k in DESIRED_RECORDING_ATT
                }
            )
        return rel_json

    for i in range(1, len(spec)):
        query += f'%20{spec[i]}'
    try:
        req = urllib.request.Request(f'{XENO_CANTO_URL}{query}&page={page}')
        response = urllib.request.urlopen(req)
        data = response.read()
        json_data = json.loads(data)
        return relevant_json(json_data)
    except Exception as ex:
        print(f'ERROR: Could not get json from xeno-canto. Reason="{ex}""')


def save_json(species: str, data: json):
    ''' Save a given JSON dict to json file '''
    page = data['page']
    with open(
        f'{SPECIES_PATH}{species}/json/{species}_{page}.json', 'w+'
    ) as output_file:
        json.dump(data, output_file, indent=4)


def main():
    args = docopt(
        """
    Usage:
        xeno_canto.py [options] <filename>
    
    Options:
        --species NUM       No. of Species
        --all-species       For all species in input file
        --audio-only        Download audio files only
        --json-only         Download json files only
    """
    )
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
            SPECIES_TO_DOWNLOAD.append(line.strip())
            line = input_file.readline()
            line_count += 1

    if args['--json-only']:
        download_json_data()

    elif args['--audio-only']:
        download_audio_data()

    else:
        download_json_data()
        download_audio_data()


if __name__ == "__main__":
    main()
