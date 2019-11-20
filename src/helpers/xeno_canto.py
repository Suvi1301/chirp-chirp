import json
import urllib.request
import copy
import sys
import os

XENO_CANTO_URL = 'https://www.xeno-canto.org/api/2/recordings?query='
DESIRED_RECORDING_ATT = ('en', 'file', 'id', 'length', 'cnt')
SPECIES_TO_DOWNLOAD = []
SPECIES_PATH = './../data/audio/raw/'


def download_data():
    def make_dir(dir: str):
        try:
            os.mkdir(dir)
        except FileExistsError as ex:
            print(f'Directory ({dir}) already exists.')
        except OSError as ex:
            print(f'ERROR: Failed to create directory {dir}. Reason="{ex}"')

    for species in SPECIES_TO_DOWNLOAD:
        make_dir(f'{SPECIES_PATH}{species}')
        make_dir(f'{SPECIES_PATH}{species}/json')
        data = get_json(species)
        save_json(species, data)
        if data['numPages'] > 1:
            i = 2
            while i <= data['numPages']:
                data = get_json(species, page=i)
                save_json(species, data)
                i += 1


def get_json(species: str, page: int = 1):
    spec = species.split('_')
    query = spec[0]

    def relevant_json(data: json):
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
    page = data['page']
    with open(
        f'{SPECIES_PATH}{species}/json/{species}_{page}.json', 'w+'
    ) as output_file:
        json.dump(data, output_file, indent=4)
    print(f'Saved {species} {page}/{data["numPages"]}')


if __name__ == "__main__":
    species_count = 1
    try:
        species_count = int(sys.argv[2])
    except TypeError as ex:
        print(f'ERROR: arg 2 must be integer')
    except IndexError as ex:
        print(f'Desired Species count not provided. Using {species_count}')

    try:
        with open(sys.argv[1]) as input_file:
            line = input_file.readline()
            line_count = 0
            while line and line_count < species_count:
                SPECIES_TO_DOWNLOAD.append(line.strip())
                line = input_file.readline()
                line_count += 1
        download_data()
    except IndexError as ex:
        print(f'ERROR: Missing input file name. Reason="{ex}"')
