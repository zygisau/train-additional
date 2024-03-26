import argparse as ag
import json


def get_parser_with_args(metadata_json='metadata.jsonc'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--max_epochs', type=int)
        parser.add_argument('--lr', type=float)
        return parser, metadata

    return None
