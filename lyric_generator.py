# SLIM
# A standalone Python script for SLIM to fine-tune a language model and generate lyrics.

import os
import re
import random
import torch
import nltk
import argparse
import pickle
import yaml
import platform
import sys
import glob
import subprocess
from ftfy import fix_text
from datasets import Dataset, DatasetDict
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)


def load_config(path='config.yaml'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Error: Configuration file not found at '{path}'.")
        print("Please create a 'config.yaml' file in the same directory.")
        sys.exit(1)


# --- Dependency Check ---
def check_dependencies(config):
    """Checks for model-specific dependencies before running."""
    if 't5' in config['model_name'].lower():
        try:
            import sentencepiece
        except ImportError:
            print("Error: The selected T5 model requires the 'sentencepiece' library.")
            print("Please install it by running: pip install sentencepiece")
            sys.exit(1)


# --- 2. DATA PREPARATION PIPELINE ---
def sanitize_raw_text(text):
    print("Step 3/7: Sanitizing raw text...")
    text = re.sub(r'\bi\.e\.\b', 'that is', text, flags=re.IGNORECASE)
    text = re.sub(r'\be\.g\.\b', 'for example', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\^(\s*\d+)?\s*', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'"', '', text)
    text = re.sub(r"[^a-zA-Z0-9\s.,?!'-]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_and_prepare_sentences(file_paths, config):
    """
    Cleans the source text from one or more files, removes publication artifacts,
    and tokenizes it into high-quality sentences using NLTK.
    """
    print("--- Starting Data Preparation ---")
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK's 'punkt' model for sentence splitting...")
        nltk.download('punkt', quiet=True)

    print(f"Step 1/7: Reading {len(file_paths)} file(s)...")
    all_lines = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_lines.extend(f.readlines())

    print("Step 2/7: Filtering publication artifacts...")
    content_lines = []
    try:
        start_index = next(
            i for i, line in enumerate(all_lines) if '*** START OF THE PROJECT GUTENBERG EBOOK' in line) + 1
        end_index = next(i for i, line in enumerate(all_lines) if '*** END OF THE PROJECT GUTENBERG EBOOK' in line)
        all_lines = all_lines[start_index:end_index]
        print("  > Found and isolated Project Gutenberg core text.")
    except StopIteration:
        print("  > Gutenberg markers not found, processing the whole file(s).")

    for line in all_lines:
        stripped_line = line.strip()
        if not stripped_line: continue
        stripped_line = re.sub(r'^\d{1,3}:\d{1,3}\s*', '', stripped_line)
        if re.match(r'^The (First|Second|Third|Fourth|Fifth|Book|Gospel|Lamentations|Song) of', stripped_line): continue
        if re.match(r'^The Book of the Prophet', stripped_line): continue
        if re.match(r'https?://\S+', stripped_line): continue
        if re.match(r'^BOOK\s+[IVXLC]+\s*\.?$', stripped_line, re.IGNORECASE): continue
        if stripped_line.isdigit(): continue
        if stripped_line.isupper() and len(stripped_line) < 40: continue
        if len(re.findall(r'[a-zA-Z]', stripped_line)) < 5: continue
        content_lines.append(stripped_line)

    raw_text = " ".join(content_lines)
    repaired_text = fix_text(raw_text)
    sanitized_text = sanitize_raw_text(repaired_text)

    print("Step 4/7: Removing remaining document artifacts...")
    text = re.sub(r'Page \d+ of \d+', '', sanitized_text)
    text = re.sub(r'Chapter \d+', '', text)

    print("Step 5/7: Splitting text into sentences...")
    sentences = nltk.sent_tokenize(text)

    print(f"Step 6/7: Filtering sentences...")
    cleaned_sentences = []
    for sentence in sentences:
        word_count = len(sentence.split())
        if config['min_words_per_sentence'] <= word_count <= config['max_words_per_sentence']:
            cleaned_sentences.append(sentence.strip().capitalize())

    print(f"Step 7/7: Preprocessing complete. Found {len(cleaned_sentences)} high-quality sentences.")
    return cleaned_sentences


def save_cleaned_sentences(sentences, output_path):
    """Saves the list of cleaned sentences to a text file for review."""
    print(f"--- Dumping sanitised text for review... ---")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(sentences))
        print(f"Successfully saved sanitised text to '{output_path}'")
    except Exception as e:
        print(f"Warning: Could not save sanitised text file. {e}")


def prepare_datasets(config, file_paths, no_cache=False):
    is_t5_model = 't5' in config['model_name'].lower()
    model_type_str = "t5" if is_t5_model else "gpt2"

    cache_dir = config['cache_dir']
    os.makedirs(cache_dir, exist_ok=True)

    sanitized_file_base = os.path.basename(config['output_dir'])
    cache_file = os.path.join(cache_dir, f"{sanitized_file_base}_{model_type_str}_tokenized.pkl")

    if not no_cache and os.path.exists(cache_file):
        print(f"--- Loading cached tokenized datasets from '{cache_file}' ---")
        with open(cache_file, 'rb') as f:
            tokenized_datasets = pickle.load(f)
    else:
        print("--- No cache found or --no-cache used. Processing data from scratch. ---")
        clean_sentences = clean_and_prepare_sentences(file_paths, config)

        # --- THE FIX IS HERE ---
        sanitized_output_path = os.path.join("sanitised_input", f"{sanitized_file_base}_sanitised.txt")
        save_cleaned_sentences(clean_sentences, sanitized_output_path)

        if is_t5_model:
            prefix = "write a lyric based on this text: "
            data_dict = {'input_text': [prefix + s for s in clean_sentences], 'target_text': clean_sentences}
        else:
            data_dict = {'text': clean_sentences}

        num_examples = len(data_dict[list(data_dict.keys())[0]])
        split_index = int(num_examples * (1 - config['validation_set_size']))
        train_dict = {k: v[:split_index] for k, v in data_dict.items()}
        eval_dict = {k: v[split_index:] for k, v in data_dict.items()}

        datasets = DatasetDict({'train': Dataset.from_dict(train_dict), 'eval': Dataset.from_dict(eval_dict)})

        TokenizerClass = T5Tokenizer if is_t5_model else GPT2Tokenizer
        tokenizer = TokenizerClass.from_pretrained(config['model_name'])

        if not is_t5_model and tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def tokenize_function(examples):
            if is_t5_model:
                model_inputs = tokenizer(examples['input_text'], max_length=128, truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(examples['target_text'], max_length=128, truncation=True)
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            else:
                return tokenizer(examples['text'], truncation=False)

        tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=list(data_dict.keys()))

        with open(cache_file, 'wb') as f:
            pickle.dump(tokenized_datasets, f)

    TokenizerClass = T5Tokenizer if is_t5_model else GPT2Tokenizer
    tokenizer = TokenizerClass.from_pretrained(config['model_name'])
    if not is_t5_model and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=config[
        'model_name']) if is_t5_model else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return tokenized_datasets['train'], tokenized_datasets['eval'], tokenizer, data_collator


# --- 3. MODEL TRAINING ---
def train_model(config, train_dataset, eval_dataset, tokenizer, data_collator):
    print("\n" + "=" * 50)
    print("               MODEL TRAINING")
    print("=" * 50 + "\n")

    is_t5_model = 't5' in config['model_name'].lower()
    ModelClass = T5ForConditionalGeneration if is_t5_model else GPT2LMHeadModel

    model = ModelClass.from_pretrained(config['model_name'])
    if not is_t5_model:
        model.resize_token_embeddings(len(tokenizer))

    steps_per_epoch = len(train_dataset) // config['batch_size']

    current_os = platform.system().lower()
    config_os = config.get("operating_system", "auto").lower()
    use_nvidia_optim = (config_os == "auto" and current_os in ["windows", "linux"]) or (
                config_os in ["windows", "linux"])

    args_dict = {
        'output_dir': config['output_dir'], 'overwrite_output_dir': True,
        'num_train_epochs': config['training_epochs'], 'per_device_train_batch_size': config['batch_size'],
        'save_steps': steps_per_epoch, 'save_total_limit': 2,
        'logging_steps': 500, 'do_eval': True, 'eval_steps': steps_per_epoch,
        'learning_rate': config['learning_rate'], 'gradient_accumulation_steps': 1,
        'torch_compile': False,
    }
    if not is_t5_model: args_dict['prediction_loss_only'] = True

    if use_nvidia_optim and torch.cuda.is_available():
        args_dict.update({'fp16': True, 'optim': "adamw_bnb_8bit"})
    else:
        args_dict.update({'fp16': False, 'dataloader_pin_memory': False})

    training_args = TrainingArguments(**args_dict)
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset,
                      eval_dataset=eval_dataset)
    trainer.train()

    print(f"Saving the final model to {config['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config['output_dir'])


# --- 4. SONG GENERATION ---
def get_initial_prompt(config, file_path, use_random=False):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception:
        return "The world is"
    if use_random:
        words = text.split()
        return ' '.join(words[random.randint(0, len(words) - 5):][:5]) if len(words) > 5 else "The world is"
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = re.split(r'(?<=[.?!])\s+', text)
    if config['use_keyword_prompt']:
        keyword_sentences = [s for s in sentences if config['prompt'].lower() in s.lower()]
        if keyword_sentences: return " ".join(random.choice(keyword_sentences).strip().split()[:20])
        print(f"Warning: Keyword not found. Using a random prompt instead.")
    return ' '.join(text.split()[:10])


def generate_section(model, tokenizer, device, prompt_text, config, original_file_path, is_t5=False, max_retries=3):
    base_temp = config['creativity_temperature']
    for attempt in range(max_retries):
        temp = base_temp + (0.1 * attempt)
        top_k = 50 + (10 * attempt)
        input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        max_len = config['tokens_per_section'] if is_t5 else len(input_ids[0]) + config['tokens_per_section']
        output_sequences = model.generate(
            input_ids=input_ids, max_length=max_len, do_sample=True,
            temperature=temp, top_k=top_k, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id, repetition_penalty=config['repetition_penalty']
        )
        newly_generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()
        if not is_t5: newly_generated_text = newly_generated_text[len(prompt_text):].strip()
        if len(newly_generated_text.split()) > 2: return newly_generated_text
        print(f"  > Generated text was too short. Retrying... (Attempt {attempt + 1}/{max_retries})")
    print("  > All retries failed. Using a random sentence from source text as a safety net.")
    try:
        with open(original_file_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f.readlines() if line.strip()]
        return random.choice(lines).strip()
    except Exception:
        return "The world keeps turning on and on."


def format_section_into_lyrical_lines(text_block, ideal_words_per_line):
    words = text_block.replace('\n', ' ').split()
    if not words: return ""
    lyrical_lines, current_line_words = [], []
    for word in words:
        current_line_words.append(word)
        if len(current_line_words) >= ideal_words_per_line:
            lyrical_lines.append(" ".join(current_line_words))
            current_line_words = []
    if current_line_words: lyrical_lines.append(" ".join(current_line_words))
    return "\n".join(lyrical_lines)


def generate_song(config, file_path, use_random_prompt=False, use_quantised=False):
    print("\n" + "=" * 50)
    print("           LYRIC COMPOSITION ENGINE")
    print("=" * 50 + "\n")

    is_t5_model = 't5' in config['model_name'].lower()
    ModelClass = T5ForConditionalGeneration if is_t5_model else GPT2LMHeadModel
    TokenizerClass = T5Tokenizer if is_t5_model else GPT2Tokenizer

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if use_quantised and device.type == 'mps':
        print("\n--- WARNING: Quantisation is not supported on Apple Silicon (MPS). ---")
        print("--- Falling back to the full-precision model for generation. ---\n")
        use_quantised = False

    model_dir_to_load = config['quantised_output_dir'] if use_quantised else config['output_dir']
    print(f"Loading model from: {model_dir_to_load}")

    if not os.path.exists(model_dir_to_load):
        sys.exit(
            f"Error: Model directory not found at '{model_dir_to_load}'. Please train or quantise the model first.")

    tokenizer = TokenizerClass.from_pretrained(model_dir_to_load)

    if use_quantised and not is_t5_model:
        print("--- Loading QUANTISED GPT-2 style model ---")
        model = ModelClass.from_pretrained(config['output_dir'])
        model.eval()
        torch.backends.quantized.engine = 'qnnpack'
        quantised_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        quantised_model.load_state_dict(torch.load(os.path.join(model_dir_to_load, "quantised_model.pt")))
        model = quantised_model
    else:
        print(f"--- Loading {'QUANTISED T5' if use_quantised else 'FULL-PRECISION'} model ---")
        model = ModelClass.from_pretrained(model_dir_to_load)

    model.to(device)

    full_song_text, chorus_text = "", ""
    initial_idea = get_initial_prompt(config, file_path, use_random=use_random_prompt)
    print(f"Starting song with initial idea: '{initial_idea}'")
    current_context = initial_idea

    for i, section_tag in enumerate(config['song_structure']):
        print(f"Composing section {i + 1}/{len(config['song_structure'])}: {section_tag}...")
        if is_t5_model:
            prompt = f"write a {section_tag.lower()} about: {current_context}"
            if "chorus" in section_tag.lower() and chorus_text:
                section_content = chorus_text
            else:
                generated_block = generate_section(model, tokenizer, device, prompt, config, file_path, is_t5=True)
                section_content = format_section_into_lyrical_lines(generated_block, config['ideal_words_per_line'])
                if "chorus" in section_tag.lower(): chorus_text = section_content
        else:
            prompt = f"{current_context}\n\n{section_tag}\n"
            if "chorus" in section_tag.lower() and chorus_text:
                section_content = chorus_text
            else:
                generated_block = generate_section(model, tokenizer, device, prompt, config, file_path)
                section_content = format_section_into_lyrical_lines(generated_block, config['ideal_words_per_line'])
                if "chorus" in section_tag.lower(): chorus_text = section_content
        full_song_text += f"\n\n{section_tag}\n{section_content}"
        if not is_t5_model: current_context = full_song_text

    final_song = full_song_text.strip()
    print("\n--- FINAL COMPOSITION ---")
    print(final_song)
    print("\n" + "=" * 50)

    if config['save_song_to_file']:
        save_song(final_song, config, file_path, initial_idea)


def save_song(song_text, config, original_file_path, initial_idea):
    print("\nSaving song to file...")
    try:
        name_without_ext = os.path.splitext(os.path.basename(original_file_path))[0]
        processed_name = re.sub(r'lyrics', '', name_without_ext, flags=re.IGNORECASE).strip().replace(' ', '_')
        model_name_folder = config['model_name'].replace("/", "_")
        song_subfolder = os.path.join(config['song_output_folder'], model_name_folder, processed_name)
        os.makedirs(song_subfolder, exist_ok=True)
        i = 1
        while True:
            song_filename = f"{processed_name}_song_{i}.txt"
            full_path = os.path.join(song_subfolder, song_filename)
            if not os.path.exists(full_path): break
            i += 1
        keys_to_save = ['model_name', 'tokens_per_section', 'ideal_words_per_line', 'creativity_temperature',
                        'learning_rate']
        config_string = "\n".join([f"{key}: {config[key]}" for key in keys_to_save if key in config])
        file_content = f"--- Generation Parameters ---\n{config_string}\n{'-' * 40}\nStarting song with initial idea: '{initial_idea}'\n{'-' * 40}\n\n{song_text}"
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        print(f"Successfully saved song to: {full_path}")
    except Exception as e:
        print(f"Error: Could not save the song. {e}")


# --- 5. MAIN EXECUTION BLOCK ---
def main():
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

    config = load_config()
    check_dependencies(config)

    for key, value in config.items():
        if isinstance(value, (int, float, str)):
            parser.add_argument(f"--{key.replace('_', '-')}", type=type(value),
                                help=f"Overrides config value for '{key}'")

    args = parser.parse_args()

    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
    config['learning_rate'] = float(config['learning_rate'])

    if os.path.isdir(args.input_path):
        file_paths = glob.glob(os.path.join(args.input_path, "*.txt"))
        if not file_paths: sys.exit(f"Error: No .txt files found in directory '{args.input_path}'")
        name_for_output = os.path.basename(os.path.normpath(args.input_path))
    elif os.path.isfile(args.input_path):
        file_paths = [args.input_path]
        name_for_output = os.path.splitext(os.path.basename(args.input_path))[0]
    else:
        sys.exit(f"Error: Input path not found at '{args.input_path}'")

    processed_name = re.sub(r'lyrics', '', name_for_output, flags=re.IGNORECASE).strip().replace(' ', '_')
    model_folder_name = config['model_name'].replace("/", "_")
    config['output_dir'] = f'./models/{model_folder_name}_{processed_name}'
    config['quantised_output_dir'] = f'./models_quantised/{model_folder_name}_{processed_name}'

    model_exists = os.path.exists(config['output_dir'])
    if not model_exists or args.force_retrain:
        if model_exists and args.force_retrain: print(
            f"--- --force-retrain flag used. Re-training model for '{args.input_path}'. ---")
        train_dataset, eval_dataset, tokenizer, data_collator = prepare_datasets(config, file_paths,
                                                                                 no_cache=args.no_cache)
        train_model(config, train_dataset, eval_dataset, tokenizer, data_collator)

    if args.quantise:
        if not os.path.exists(config['quantised_output_dir']):
            print(f"--- Quantised model not found. Running quantisation script... ---")
            quantise_script_path = config.get('quantise_script_path', './extra/quantise_model.py')
            if not os.path.exists(quantise_script_path):
                sys.exit(f"Error: Quantisation script not found at '{quantise_script_path}'.")
            command = [sys.executable, quantise_script_path, config['output_dir'], "-o",
                       os.path.dirname(config['quantised_output_dir'])]
            result = subprocess.run(command)
            if result.returncode != 0: sys.exit("Error: Quantisation script failed to execute.")

    generate_song(config, file_paths[0], use_random_prompt=args.random, use_quantised=args.quantise)


if __name__ == "__main__":
    main()
