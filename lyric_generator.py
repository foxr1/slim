import glob
import logging
import os
import re
import sys

from generator.data_pipeline import DataPipeline
from generator.lyric_model import LyricModel
from generator.utils import load_config, parse_arguments


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    try:
        config = load_config()
        args = parse_arguments(config)

        for key, value in vars(args).items():
            if value is not None and key in config:
                config[key] = value

        config['learning_rate'] = float(config['learning_rate'])

        if os.path.isdir(args.input_path):
            file_paths = glob.glob(os.path.join(args.input_path, "*.txt"))
            if not file_paths:
                logging.error(f"No .txt files found in directory '{args.input_path}'")
                sys.exit(1)
            name_for_output = os.path.basename(os.path.normpath(args.input_path))
        elif os.path.isfile(args.input_path):
            file_paths = [args.input_path]
            name_for_output = os.path.splitext(os.path.basename(args.input_path))[0]
        else:
            logging.error(f"Input path not found at '{args.input_path}'")
            sys.exit(1)

        logging.info(f"Processing input: {args.input_path}")

        processed_name = re.sub(r'lyrics', '', name_for_output, flags=re.IGNORECASE).strip().replace(' ', '_')
        model_folder_name = config['model_name'].replace("/", "_")
        config['output_dir'] = f'./models/{model_folder_name}_{processed_name}'
        config['quantised_output_dir'] = f'./models_quantised/{model_folder_name}_{processed_name}'

        pipeline = DataPipeline(config)
        model_manager = LyricModel(config)

        model_exists = os.path.exists(config['output_dir'])

        if not model_exists or args.force_retrain:
            if model_exists and args.force_retrain:
                logging.info(f"--force-retrain flag used. Re-training model")

            train_dataset, eval_dataset, tokenizer, data_collator = pipeline.prepare_datasets(file_paths, no_cache=args.no_cache)
            model_manager.train(train_dataset, eval_dataset, tokenizer, data_collator)
        else:
            logging.info(f"Found existing model at '{config['output_dir']}'. Skipping training")
            logging.info("    (Use the --force-retrain flag to train again from scratch.)")

        # Generate Song
        model_manager.generate(
            file_paths[0],
            use_random_prompt=args.random,
            use_quantised=args.quantise
        )

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()