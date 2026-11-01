import argparse
import json
from pathlib import Path


def process(items):
    values = list(items)
    next_values = []
    for value in values:
        if value > 0:
            next_values.append(value)
    values = next_values
    next_values = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            next_values.append(value)
    values = next_values
    values = sorted(values)
    return values


def main():
    parser = argparse.ArgumentParser(
        description='Process a JSON list of integer readings.')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output files must differ.')
    try:
        values = json.loads(args.input.read_text())
        if not isinstance(values, list) or any(
                type(value) is not int for value in values):
            raise ValueError('Expected a JSON list of integers.')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(process(values)) + '\n')


if __name__ == '__main__':
    main()
