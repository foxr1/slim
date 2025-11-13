# Automates running multiple training jobs for SLIM

import os
import sys
import subprocess
import logging
import argparse
import time
import pandas as pd
from tqdm import tqdm
from plotnine import ggplot, aes, geom_bar, labs, theme, element_text, ggsave

MODELS_TO_TEST = [
    "gpt2-medium",
    "distilbert/distilgpt2",
    "facebook/opt-350m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen3-0.6B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
]

INPUT_FILES = [
    "input_files/harry/Frankenstein.txt",
    "input_files/harry/The Sun Also Rises.txt",
    "input_files/harry/The Wit and Humour of America, Volume 1.txt"
]

CONFIG_PATH = "config.yaml"
SCRIPT_NAME = "lyric_generator.py"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def run_training_job(model_name, input_file, verbose=False, tqdm_iterator=None):
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
    if not verbose:
        command.append("--no-progress-bar")

    if verbose:
        logging.info(f"Executing command: {' '.join(command)}")

    process = None
    start_time = time.time()
    try:
        if verbose:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', bufsize=1, universal_newlines=True)
            for line in process.stdout:
                print(line, end='')
            process.wait()
            stderr_output = None
        else:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                       text=True, encoding='utf-8')
            _, stderr_output = process.communicate()

        end_time = time.time()
        duration = end_time - start_time

        if process.returncode == 0:
            if verbose:
                logging.info(f"JOB SUCCEEDED: {job_id} in {duration:.2f} seconds")
            return {'model': model_name, 'input_file': os.path.basename(input_file), 'time': duration, 'status': 'success'}
        else:
            msg = f"JOB FAILED: {job_id} (Return Code: {process.returncode})"
            if tqdm_iterator:
                tqdm_iterator.write(msg)
                if stderr_output:
                    tqdm_iterator.write(stderr_output.strip())
            else:
                logging.error(msg)
                if stderr_output:
                    logging.error(stderr_output.strip())
            return {'model': model_name, 'input_file': os.path.basename(input_file), 'time': duration, 'status': 'failed'}

    except subprocess.CalledProcessError as e:
        logging.error(f"JOB FAILED: {job_id}")
        logging.error(f"Error details: {e}")
        return {'model': model_name, 'input_file': os.path.basename(input_file), 'time': time.time() - start_time, 'status': 'failed'}
    except KeyboardInterrupt:
        logging.warning("Job interrupted by user.")
        if process:
            process.terminate()
        raise

def create_visualisation(results_df):
    if results_df.empty:
        logging.warning("No results to visualise.")
        return

    p = (ggplot(results_df, aes(x='input_file', y='time', fill='model'))
         + geom_bar(stat='identity', position='dodge')
         + labs(title='Training Time by Model and Input File',
                x='Input File',
                y='Training Time (seconds)',
                fill='Model')
         + theme(axis_text_x=element_text(rotation=45, hjust=1)))
    
    output_filename = "training_visualisation.png"
    ggsave(p, filename=output_filename, dpi=300)
    logging.info(f"Visualisation saved to {output_filename}")

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

    results = []
    try:
        for model_name, input_file in job_iterator:
            if not args.verbose:
                job_iterator.set_description(f"Running: {model_name.split('/')[-1]}")

            result = run_training_job(model_name, input_file, verbose=args.verbose,
                                      tqdm_iterator=job_iterator if not args.verbose else None)
            if result:
                results.append(result)

    except KeyboardInterrupt:
        logging.info("\nBatch run aborted by user.")
    finally:
        logging.info("Batch run finished")
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv("training_results.csv", index=False)
            logging.info("Training results saved to training_results.csv")
            create_visualisation(results_df)
        else:
            logging.warning("No results to save or visualise.")


if __name__ == "__main__":
    main()
