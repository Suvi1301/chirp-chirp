import os

from docopt import docopt
from progress.bar import Bar

IMAGES_PATH = './../data/images/'


def _make_dir(dir: str):
    ''' Create a new directory if not already exists '''
    try:
        os.mkdir(dir)
    except FileExistsError as ex:
        print(f'Directory ({dir}) already exists.')
    except OSError as ex:
        print(f'ERROR: Failed to create directory {dir}. Reason="{ex}"')


def copy(species: list, train_size: int = 5000, test_size: int = 1000):
    for spec in species:
        _make_dir(f'{IMAGES_PATH}training_data/{spec}')
        _make_dir(f'{IMAGES_PATH}testing_data/{spec}')
        i = 1
        with Bar(
            f'Copying Training and Testing dataset for {spec}',
            suffix='%(percent)d%%',
            max=train_size + test_size,
        ) as bar:
            for file in os.listdir(f'{IMAGES_PATH}spectrograms/{spec}'):
                try:
                    if file.endswith('.jpg') and i <= train_size:
                        os.system(
                            f'cp {IMAGES_PATH}spectrograms/{spec}/{file} {IMAGES_PATH}training_data/{spec}/{file}'
                        )
                        i += 1
                    else:
                        os.system(
                            f'cp {IMAGES_PATH}spectrograms/{spec}/{file} {IMAGES_PATH}testing_data/{spec}/{file}'
                        )
                        i += 1
                    bar.next()
                    if i >= train_size + test_size:
                        break
                    bar.next()
                except Exception as ex:
                    print(f'ERROR: Cannot copy file. Reason="{ex}"')


def main():
    args = docopt(
        """
    Usage:
        cnn_helper.py [options] <filename>

    Options:
        --species NUM       No. of Species
        --all-species       For all species in input file
        --train-size NUM    Training set size.
        --test-size NUM    Testing set size.
    """
    )
    species = []
    train_size = 5000
    test_size = 1000

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
            species.append(line.strip())
            line = input_file.readline()
            line_count += 1

    if args['--train-size']:
        try:
            train_size = int(args['--train-size'])
        except TypeError:
            print(f'ERROR: --train-size must be an integer')

    if args['--test-size']:
        try:
            train_size = int(args['--test-size'])
        except TypeError:
            print(f'ERROR: --test-size must be an integer')

    print(f'Generate Training and Testing data')
    copy(species, train_size=train_size, test_size=test_size)


if __name__ == "__main__":
    main()
