"""
Tests several datasets and store the results.
"""

import sys, os
import json

__author__ = 'Henry Cagnini'


def main():
    datasets_path = '/home/henry/Projects/eel/datasets/uci'

    params_path = '/home/henry/Projects/eel/params.json'
    reporter_output = "/home/henry/Projects/eel/metadata"

    datasets = os.listdir(datasets_path)
    params_file = json.load(open(params_path))

    for dataset in datasets:
        dataset_name = dataset.split('.')[0]

        print 'testing %s dataset' % dataset_name

        params_file['full_path'] = os.path.join(datasets_path, dataset)
        params_file['reporter_output'] = os.path.join(reporter_output, dataset_name)

        os.mkdir(params_file['reporter_output'])

        json.dump(params_file, open(params_path, 'w'), indent=2)

        variables = {}
        sys.stdout = open(
            os.path.join(params_file['reporter_output'], dataset_name + '.txt'), 'w'
        )
        try:
            execfile('/home/henry/Projects/eel/__main__.py', variables)
        except Exception as e:
            with open(os.path.join(params_file['reporter_output'], 'exception.txt'), 'w') as f:
                f.write(str(e.message) + '\n' + str(e.args))


if __name__ == '__main__':
    main()
