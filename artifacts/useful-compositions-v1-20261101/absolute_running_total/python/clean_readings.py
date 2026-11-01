import argparse
import json
from pathlib import Path


def process(items: list[int]) -> list[int]:
    values = list(items)
    next_values = []
    for value in values:
        next_values.append(abs(value))
    values = next_values
    next_values = []
    total = 0
    for value in values:
        total += value
        next_values.append(total)
    values = next_values
    return values


def main() -> None:
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
