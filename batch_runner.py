# Automates running multiple training jobs for SLIM

import os
import sys
import subprocess
import logging
import argparse
from tqdm import tqdm

MODELS_TO_TEST = [
    "gpt2-medium",
    "distilbert/distilgpt2",
    "facebook/opt-350m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-3B",
    "mistralai/Mistral-7B-Instruct-v0.2"
]

INPUT_FILES = [
    "input_files/Big Trousers lyrics.txt"
]

CONFIG_PATH = "config.yaml"
SCRIPT_NAME = "lyric_generator.py"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def run_training_job(model_name, input_file, verbose=False):
    job_id = f"Model='{model_name}', Input='{input_file}'"

    if verbose:
        logging.info("=" * 80)
        logging.info(f"STARTING JOB: {job_id}")
        logging.info("=" * 80)

    command = [
        sys.executable,
        SCRIPT_NAME,
        input_file,
        "--force-retrain",
        "--model-name", model_name
    ]

    if verbose:
        logging.info(f"Executing command: {' '.join(command)}")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8')

        if verbose:
            for line in process.stdout:
                print(line, end='')
        else:
            for _ in process.stdout:
                pass

        process.wait()

        if process.returncode == 0:
            if verbose:
                logging.info(f"JOB SUCCEEDED: {job_id}")
        else:
            logging.error(f"JOB FAILED: {job_id} (Return Code: {process.returncode})")

    except subprocess.CalledProcessError as e:
        logging.error(f"JOB FAILED: {job_id}")
        logging.error(f"Error details: {e}")
    except KeyboardInterrupt:
        logging.warning("Job interrupted by user.")
        process.terminate()
        raise


def main():
    parser = argparse.ArgumentParser(description="Batch runner for SLIM.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show the full script output instead of the overall progress bar.")
    args = parser.parse_args()

    logging.info("Starting Batch Training Run")

    jobs = []
    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            logging.warning(f"Input file '{input_file}' not found. Skipping.")
            continue
        for model_name in MODELS_TO_TEST:
            jobs.append((model_name, input_file))

    logging.info(f"Models to test: {MODELS_TO_TEST}")
    logging.info(f"Input files: {INPUT_FILES}")
    logging.info(f"Total jobs: {len(jobs)}")

    if args.verbose:
        job_iterator = jobs
        logging.info("Verbose mode enabled. Full script output will be shown.")
    else:
        job_iterator = tqdm(jobs, desc="Overall Batch Progress", unit="job", file=sys.stdout)
        logging.info("Progress bar enabled. Script output will be suppressed.")

    try:
        for model_name, input_file in job_iterator:
            if not args.verbose:
                job_iterator.set_description(f"Running: {model_name.split('/')[-1]}")

            run_training_job(model_name, input_file, verbose=args.verbose)

    except KeyboardInterrupt:
        logging.info("\nBatch run aborted by user.")
    finally:
        logging.info("Batch run finished")


if __name__ == "__main__":
    main()