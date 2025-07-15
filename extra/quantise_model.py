# Quantise Model
# A utility script to apply dynamic quantisation to a fine-tuned model,
# significantly reducing its size for easier deployment. (I.e. on the web)

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def quantise_model(input_model_path, output_model_path):
    print(f"--- Starting quantisation for model at '{input_model_path}' ---")

    if not os.path.isdir(input_model_path):
        print(f"Error: Input model directory not found at '{input_model_path}'")
        sys.exit(1)

    # 2. GPT-2/OPT vs T5
    is_t5_model = 't5' in input_model_path.lower()
    if is_t5_model:
        print("Error: Quantisation for T5 models is not supported by this script.")
        sys.exit(1)

    ModelClass = AutoModelForCausalLM

    print("Loading full-precision model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(input_model_path)
        model = ModelClass.from_pretrained(input_model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Applying dynamic quantisation...")

    # Set the quantization engine to 'qnnpack' for broad compatibility
    torch.backends.quantized.engine = 'qnnpack'
    quantised_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    print(f"Saving quantised model to '{output_model_path}'...")
    os.makedirs(output_model_path, exist_ok=True)

    torch.save(quantised_model.state_dict(), os.path.join(output_model_path, "quantised_model.pt"))
    tokenizer.save_pretrained(output_model_path)
    model.config.save_pretrained(output_model_path)

    print("\n--- Quantisation Complete ---")
    original_size = sum(
        os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(input_model_path) for
        filename in filenames)
    quantised_size = sum(
        os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(output_model_path) for
        filename in filenames)

    print(f"Original model size: {original_size / 1e6:.2f} MB")
    print(f"Quantised model size: {quantised_size / 1e6:.2f} MB")
    print(f"Size reduction: {100 * (1 - quantised_size / original_size):.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Quantise a fine-tuned Hugging Face model for deployment."
    )

    parser.add_argument(
        "input_model_path",
        metavar="INPUT_MODEL_DIR",
        type=str,
        help="Path to the directory containing the fine-tuned model."
    )

    parser.add_argument(
        "-o", "--output-dir",
        dest="output_base_dir",
        type=str,
        default="./models_quantised",
        help="The base directory where the quantised model folder will be saved. Defaults to './models_quantised'."
    )

    args = parser.parse_args()

    model_name = os.path.basename(args.input_model_path)
    output_model_path = os.path.join(args.output_base_dir, model_name)

    quantise_model(args.input_model_path, output_model_path)


if __name__ == "__main__":
    main()
