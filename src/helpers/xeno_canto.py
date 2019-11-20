import json
import urllib
import copy
import sys

XENO_CANTO_URL = 'https://www.xeno-canto.org/api/2/recordings?query='
DESIRED_RECORDING_ATT = ('en', 'file', 'id', 'length', 'cnt')
SPECIES_TO_DOWNLOAD = []
JSON_DATA_PATH = '../data/raw/json/'


def download_data():
    pass


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
        print(f'ERROR: getting json from xeno-canto. Reason="{ex}""')


def save_json(species: str, data: json, page: int):
    with open(f'{JSON_DATA_PATH}{species}_{page}.json', 'w') as output_file:
        json.dump(data, output_file)


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
