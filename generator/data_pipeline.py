# This class handles all data loading, cleaning, processing, and caching.

import os
import re
import nltk
import pickle
import logging
from ftfy import fix_text
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling
)


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.model_name = config['model_name']

    def _sanitize_raw_text(self, text):
        logging.info("Step 3/7: Sanitizing raw text...")
        text = re.sub(r'\bi\.e\.\b', 'that is', text, flags=re.IGNORECASE)
        text = re.sub(r'\be\.g\.\b', 'for example', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\^(\s*\d+)?\s*', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'"', '', text)
        text = re.sub(r"[^a-zA-Z0-9\s.,?!'-]", '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _clean_and_prepare_sentences(self, file_paths):
        logging.info("Starting Data Preparation")
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            logging.info("Downloading NLTK's 'punkt' model for sentence splitting...")
            nltk.download('punkt', quiet=True)

        logging.info(f"Step 1/7: Reading {len(file_paths)} file(s)...")
        all_lines = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_lines.extend(f.readlines())

        logging.info("Step 2/7: Filtering publication artifacts...")
        content_lines = []
        try:
            start_index = next(
                i for i, line in enumerate(all_lines) if '*** START OF THE PROJECT GUTENBERG EBOOK' in line) + 1
            end_index = next(i for i, line in enumerate(all_lines) if '*** END OF THE PROJECT GUTENBERG EBOOK' in line)
            all_lines = all_lines[start_index:end_index]
            logging.info("  > Found and isolated Project Gutenberg core text.")
        except StopIteration:
            logging.info("  > Gutenberg markers not found, processing the whole file(s).")

        for line in all_lines:
            stripped_line = line.strip()
            if not stripped_line: continue
            stripped_line = re.sub(r'^\d{1,3}:\d{1,3}\s*', '', stripped_line)
            stripped_line = re.sub(r'\d\n\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+', '', stripped_line)
            stripped_line = re.sub(r'https?://\S+', '', stripped_line)
            stripped_line = re.sub(r'…see more', '', stripped_line)
            if re.match(r'^The (First|Second|Third|Fourth|Fifth|Book|Gospel|Lamentations|Song) of',
                        stripped_line): continue
            if re.match(r'^The Book of the Prophet', stripped_line): continue
            if re.match(r'^BOOK\s+[IVXLC]+\s*\.?$', stripped_line, re.IGNORECASE): continue
            if stripped_line.isdigit(): continue
            if stripped_line.isupper() and len(stripped_line) < 40: continue
            if len(re.findall(r'[a-zA-Z]', stripped_line)) < 5: continue

            content_lines.append(stripped_line)

        raw_text = " ".join(content_lines)
        repaired_text = fix_text(raw_text)
        sanitized_text = self._sanitize_raw_text(repaired_text)

        logging.info("Step 4/7: Removing remaining document artifacts...")
        text = re.sub(r'Page \d+ of \d+', '', sanitized_text)
        text = re.sub(r'Chapter \d+', '', text)

        logging.info("Step 5/7: Splitting text into sentences...")
        sentences = nltk.sent_tokenize(text)

        logging.info(f"Step 6/7: Filtering sentences...")
        cleaned_sentences = []
        for sentence in sentences:
            word_count = len(sentence.split())
            if self.config['min_words_per_sentence'] <= word_count <= self.config['max_words_per_sentence']:
                cleaned_sentences.append(sentence.strip().capitalize())

        logging.info(f"Step 7/7: Preprocessing complete. Found {len(cleaned_sentences)} high-quality sentences.")
        return cleaned_sentences

    def _save_cleaned_sentences(self, sentences, output_path):
        logging.info(f"Dumping sanitised text for review..")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(sentences))
            logging.info(f"Successfully saved sanitised text to '{output_path}'")
        except Exception as e:
            logging.warning(f"Could not save sanitised text file. {e}")

    def prepare_datasets(self, file_paths, no_cache=False):
        model_type_str = "decoder"

        cache_dir = self.config['cache_dir']
        os.makedirs(cache_dir, exist_ok=True)

        sanitized_file_base = os.path.basename(self.config['output_dir'])
        cache_file = os.path.join(cache_dir, f"{sanitized_file_base}_{model_type_str}_tokenized.pkl")

        if not no_cache and os.path.exists(cache_file):
            logging.info(f"Loading cached tokenized datasets from '{cache_file}'")
            with open(cache_file, 'rb') as f:
                tokenized_datasets = pickle.load(f)
        else:
            logging.info("No cache found or --no-cache used. Processing data from scratch")
            clean_sentences = self._clean_and_prepare_sentences(file_paths)

            sanitized_output_path = os.path.join("sanitised_input", f"{sanitized_file_base}_sanitised.txt")
            self._save_cleaned_sentences(clean_sentences, sanitized_output_path)

            data_dict = {'text': clean_sentences}

            num_examples = len(data_dict[list(data_dict.keys())[0]])
            split_index = int(num_examples * (1 - self.config['validation_set_size']))
            train_dict = {k: v[:split_index] for k, v in data_dict.items()}
            eval_dict = {k: v[split_index:] for k, v in data_dict.items()}

            datasets = DatasetDict({'train': Dataset.from_dict(train_dict), 'eval': Dataset.from_dict(eval_dict)})

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            if tokenizer.pad_token is None:
                logging.info("No pad token found. Setting to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token

            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=False)

            tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=list(data_dict.keys()))

            logging.info(f"Saving tokenized datasets to cache at '{cache_file}'")
            with open(cache_file, 'wb') as f:
                pickle.dump(tokenized_datasets, f)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        return tokenized_datasets['train'], tokenized_datasets['eval'], tokenizer, data_collator