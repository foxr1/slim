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

    def _sanitise_raw_text(self, text):
        logging.info("Step 3/7: Sanitising raw text...")
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
            nltk.data.find('tokenisers/punkt')
        except LookupError:
            logging.info("Downloading NLTK's 'punkt' model for sentence splitting...")
            nltk.download('punkt', quiet=True)

        logging.info(f"Step 1/7: Reading {len(file_paths)} file(s)...")
        all_lines = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_lines.extend(f.readlines())

        if self.config.get('sanitise', True):
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
                # Remove timestamps like "0:00" or "0:00:00,000 --> 0:00:00,000"
                stripped_line = re.sub(r'^\d{1,3}:\d{1,3}\s*', '', stripped_line)
                stripped_line = re.sub(r'\d\n\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+', '', stripped_line)
                # Remove URLs
                stripped_line = re.sub(r'https?://\S+', '', stripped_line)
                # Remove "…see more"
                stripped_line = re.sub(r'…see more', '', stripped_line)
                # Skip lines that are likely book/chapter titles (e.g., "The First Book of...")
                if re.match(r'^The (First|Second|Third|Fourth|Fifth|Book|Gospel|Lamentations|Song) of',
                            stripped_line): continue
                # Skip lines: 'The Book of the Prophet'
                if re.match(r'^The Book of the Prophet', stripped_line): continue
                # Skip lines that are likely Roman numeral chapter headings (e.g., "BOOK I.")
                if re.match(r'^BOOK\s+[IVXLC]+\s*\.?$', stripped_line, re.IGNORECASE): continue
                # Skip lines that contain only digits
                if stripped_line.isdigit(): continue
                # Skip lines that are all uppercase and relatively short (likely headings or noise)
                if stripped_line.isupper() and len(stripped_line) < 40: continue
                # Skip lines with very few alphabetic characters (likely noise or non-text)
                if len(re.findall(r'[a-zA-Z]', stripped_line)) < 5: continue

                content_lines.append(stripped_line)

            raw_text = " ".join(content_lines)
            repaired_text = fix_text(raw_text)
            sanitised_text = self._sanitise_raw_text(repaired_text)

            logging.info("Step 4/7: Removing remaining document artefacts...")
            text = re.sub(r'Page \d+ of \d+', '', sanitised_text)
            text = re.sub(r'Chapter \d+', '', text)
        else:
            logging.info("Skipping sanitisation as per --sanitise=False.")
            text = " ".join(all_lines)

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

        sanitised_file_base = os.path.basename(self.config['output_dir'])
        cache_file = os.path.join(cache_dir, f"{sanitised_file_base}_{model_type_str}_tokenised.pkl")

        if not no_cache and os.path.exists(cache_file):
            logging.info(f"Loading cached tokenised datasets from '{cache_file}'")
            with open(cache_file, 'rb') as f:
                tokenised_datasets = pickle.load(f)
        else:
            logging.info("No cache found or --no-cache used. Processing data from scratch")
            clean_sentences = self._clean_and_prepare_sentences(file_paths)

            if self.config.get('sanitise', True):
                sanitised_output_path = os.path.join("sanitised_input", f"{sanitised_file_base}_sanitised.txt")
                self._save_cleaned_sentences(clean_sentences, sanitised_output_path)

            data_dict = {'text': clean_sentences}

            num_examples = len(data_dict[list(data_dict.keys())[0]])
            split_index = int(num_examples * (1 - self.config['validation_set_size']))
            train_dict = {k: v[:split_index] for k, v in data_dict.items()}
            eval_dict = {k: v[split_index:] for k, v in data_dict.items()}

            datasets = DatasetDict({'train': Dataset.from_dict(train_dict), 'eval': Dataset.from_dict(eval_dict)})

            tokeniser = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            if tokeniser.pad_token is None:
                logging.info("No pad token found. Setting to eos_token.")
                tokeniser.pad_token = tokeniser.eos_token

            def tokenise_function(examples):
                return tokeniser(examples['text'], truncation=False)

            tokenised_datasets = datasets.map(tokenise_function, batched=True, remove_columns=list(data_dict.keys()))

            logging.info(f"Saving tokenised datasets to cache at '{cache_file}'")
            with open(cache_file, 'wb') as f:
                pickle.dump(tokenised_datasets, f)

        tokeniser = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokeniser.pad_token is None:
            tokeniser.pad_token = tokeniser.eos_token

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokeniser, mlm=False)
        return tokenised_datasets['train'], tokenised_datasets['eval'], tokeniser, data_collator
