# Automates running multiple training jobs for SLIM

import os
import sys
import subprocess
import logging
import argparse
import time
import pandas as pd
from tqdm import tqdm

MODELS_TO_TEST = [
    "gpt2-medium",
    "distilbert/distilgpt2",
    "facebook/opt-350m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen3-0.6B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
]

INPUT_FILES = [
    "input_files/Frankenstein.txt",
    "input_files/The Sun Also Rises.txt",
    "input_files/The Wit and Humor of America, Volume 1.txt"
]

CREATIVITY_TEMPERATURES = [0.5, 0.75, 1.0, 1.25]
BATCH_SIZES = [4, 8]
TRAINING_EPOCHS = [1, 2, 3]
GRADIENT_ACCUMULATION_STEPS = [1, 2]
SANITISES = [True, False]
NUM_ITERATIONS = 1

CONFIG_PATH = "config.yaml"
SCRIPT_NAME = "lyric_generator.py"
RESULTS_DIR = "results"
PLOT_SCRIPT_NAME = "plot_results.py"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def run_training_job(model_name, input_file, temp, batch_size, epochs, grad_steps, sanitise, iteration, verbose=False, tqdm_iterator=None):
    job_id = (f"Model='{model_name}', Input='{input_file}', Temp='{temp}', "
              f"Batch='{batch_size}', Epochs='{epochs}', GradSteps='{grad_steps}', "
              f"Sanitise='{sanitise}', Iteration='{iteration}'")

    if verbose:
        logging.info("=" * 80)
        logging.info(f"STARTING JOB: {job_id}")
        logging.info("=" * 80)

    command = [
        sys.executable,
        SCRIPT_NAME,
        input_file,
        "--force-retrain",
        "--model-name", model_name,
        "--creativity-temperature", str(temp),
        "--batch-size", str(batch_size),
        "--training-epochs", str(epochs),
        "--gradient-accumulation-steps", str(grad_steps),
        "--sanitise", str(sanitise)
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

        result_data = {
            'model': model_name,
            'input_file': os.path.basename(input_file),
            'creativity_temperature': temp,
            'batch_size': batch_size,
            'training_epochs': epochs,
            'gradient_accumulation_steps': grad_steps,
            'sanitise': sanitise,
            'iteration': iteration,
            'time': duration,
        }

        if process.returncode == 0:
            if verbose:
                logging.info(f"JOB SUCCEEDED: {job_id} in {duration:.2f} seconds")
            result_data['status'] = 'success'
            return result_data
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
            result_data['status'] = 'failed'
            return result_data

    except subprocess.CalledProcessError as e:
        logging.error(f"JOB FAILED: {job_id}")
        logging.error(f"Error details: {e}")
        return {'model': model_name, 'input_file': os.path.basename(input_file), 'time': time.time() - start_time, 'status': 'failed'}
    except KeyboardInterrupt:
        logging.warning("Job interrupted by user.")
        if process:
            process.terminate()
        raise

def main():
    parser = argparse.ArgumentParser(description="Batch runner for SLIM.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show the full script output instead of the overall progress bar.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    logging.info("Starting Batch Training Run")

    jobs = []
    for i in range(NUM_ITERATIONS):
        for input_file in INPUT_FILES:
            if not os.path.exists(input_file):
                logging.warning(f"Input file '{input_file}' not found. Skipping.")
                continue
            for model_name in MODELS_TO_TEST:
                for temp in CREATIVITY_TEMPERATURES:
                    for batch_size in BATCH_SIZES:
                        for epochs in TRAINING_EPOCHS:
                            for grad_steps in GRADIENT_ACCUMULATION_STEPS:
                                for sanitise in SANITISES:
                                    jobs.append((model_name, input_file, temp, batch_size, epochs, grad_steps, sanitise, i + 1))

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
        for job_params in job_iterator:
            model_name, input_file, temp, batch_size, epochs, grad_steps, sanitise, iteration = job_params
            if not args.verbose:
                job_iterator.set_description(f"Running: {model_name.split('/')[-1]} (Iter {iteration})")

            result = run_training_job(model_name, input_file, temp, batch_size, epochs, grad_steps, sanitise, iteration,
                                      verbose=args.verbose,
                                      tqdm_iterator=job_iterator if not args.verbose else None)
            if result:
                results.append(result)

    except KeyboardInterrupt:
        logging.info("\nBatch run aborted by user.")
    finally:
        logging.info("Batch run finished")
        if results:
            results_df = pd.DataFrame(results)
            csv_path = os.path.join(RESULTS_DIR, "training_results.csv")
            results_df.to_csv(csv_path, index=False)
            logging.info(f"Training results saved to {csv_path}")

            logging.info("Generating visualisations...")
            subprocess.run([sys.executable, PLOT_SCRIPT_NAME, csv_path])
        else:
            logging.warning("No results to save or visualise.")


if __name__ == "__main__":
    main()
