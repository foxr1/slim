# Contains helper functions for configuration and argument parsing.

import os
import yaml
import argparse
import logging
import sys

def load_config(path='config.yaml'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logging.error(f"Configuration file not found at '{path}'.")
        sys.exit(1)


def parse_arguments(config):
    parser = argparse.ArgumentParser(description="AI Lyric Generator: Fine-tune a model and generate lyrics.")
    parser.add_argument("input_path", type=str, help="Path to the input text file or a folder containing .txt files.")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force re-training of the model, even if one already exists.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-processing of the data, ignoring any existing cache.")
    parser.add_argument("--random", action="store_true",
                        help="Use a completely random prompt instead of a keyword-based one.")
    parser.add_argument("--quantise", action="store_true",
                        help="Use a quantised model for generation, creating one if it doesn't exist.")

    for key, value in config.items():
        if isinstance(value, (int, float, str)):
            parser.add_argument(f"--{key.replace('_', '-')}", type=type(value),
                                help=f"Overrides config value for '{key}'")

    return parser.parse_args()