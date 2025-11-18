# This class handles all model-specific logic: training, generation, and loading.

import os
import re
import random
import torch
import nltk
import logging
import platform
import subprocess
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)


class LyricModel:
    def __init__(self, config):
        logging.info("Initialising LyricModel...")
        self.config = config
        self.model_name = config['model_name']
        self.device = self._get_device()
        self.tokeniser = None
        self.model = None
        self._check_dependencies()

    def _get_device(self):
        if torch.backends.mps.is_available():
            logging.info("MPS backend (Apple Silicon) detected.")
            return torch.device("mps")
        if torch.cuda.is_available():
            logging.info("CUDA backend (NVIDIA GPU) detected.")
            return torch.device("cuda")
        logging.info("No compatible GPU detected. Using CPU.")
        return torch.device("cpu")

    def _check_dependencies(self):
        current_os = platform.system().lower()
        config_os = self.config.get("operating_system", "auto").lower()
        use_nvidia_optim = (config_os == "auto" and current_os in ["windows", "linux"]) or (
                config_os in ["windows", "linux"])

        if use_nvidia_optim:
            try:
                import bitsandbytes
            except ImportError:
                logging.warning("For optimal performance on NVIDIA GPUs, install the 'bitsandbytes' library.")
                logging.warning("You can do so by running: pip install bitsandbytes")

    def train(self, train_dataset, eval_dataset, tokeniser, data_collator):
        logging.info("=" * 50)
        logging.info("               MODEL TRAINING")
        logging.info("=" * 50)

        model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        model.resize_token_embeddings(len(tokeniser))

        steps_per_epoch = len(train_dataset) // self.config['batch_size']
        if steps_per_epoch == 0:
            steps_per_epoch = 1

        current_os = platform.system().lower()
        config_os = self.config.get("operating_system", "auto").lower()
        use_nvidia_optim = (config_os == "auto" and current_os in ["windows", "linux"]) or (
                config_os in ["windows", "linux"])

        args_dict = {
            'output_dir': self.config['output_dir'], 'overwrite_output_dir': True,
            'num_train_epochs': self.config['training_epochs'],
            'per_device_train_batch_size': self.config['batch_size'],
            'save_steps': steps_per_epoch, 'save_total_limit': 2,
            'logging_steps': max(1, steps_per_epoch // 10),
            'do_eval': True, 'eval_steps': steps_per_epoch,
            'learning_rate': self.config['learning_rate'],
            'gradient_accumulation_steps': self.config['gradient_accumulation_steps'],
            'torch_compile': False,
            'prediction_loss_only': True,
            'disable_tqdm': self.config.get('no_progress_bar', False)
        }

        if use_nvidia_optim and torch.cuda.is_available():
            logging.info("Applying NVIDIA-specific optimisations (fp16, bitsandbytes)")
            args_dict.update({'fp16': True, 'optim': "adamw_bnb_8bit"})
        elif self.device.type == 'mps':
            logging.info("Applying Apple MPS optimisations")
            args_dict.update({'fp16': False, 'dataloader_pin_memory': False})
        else:
            logging.info("Using standard CPU optimisations")
            args_dict.update({'fp16': False, 'dataloader_pin_memory': False})

        training_args = TrainingArguments(**args_dict)
        trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset,
                          eval_dataset=eval_dataset)

        logging.info(f"Starting model training with '{self.model_name}' for {self.config['training_epochs']} epochs...")
        trainer.train()
        logging.info("Training finished.")

        logging.info(f"Saving the final model to {self.config['output_dir']}")
        trainer.save_model()
        tokeniser.save_pretrained(self.config['output_dir'])

    def _get_initial_prompt(self, file_path, use_random=False):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            return "The world is"

        if use_random:
            words = text.split()
            if len(words) > 5:
                start_index = random.randint(0, len(words) - 5)
                return ' '.join(words[start_index: start_index + 5])
            else:
                return "The world is"

        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.?!])\s+', text)

        if self.config['use_keyword_prompt']:
            keyword = self.config['prompt']
            keyword_sentences = [s for s in sentences if keyword.lower() in s.lower()]
            if keyword_sentences:
                return " ".join(random.choice(keyword_sentences).strip().split()[:20])
            logging.warning(f"Keyword '{keyword}' not found. Using a random prompt instead.")

        words = text.split()
        if len(words) > 10:
            start_index = random.randint(0, len(words) - 10)
            return ' '.join(words[start_index: start_index + 10])
        else:
            return "The world is"

    def _generate_section(self, prompt_text, original_file_path, max_retries=3):
        base_temp = self.config['creativity_temperature']
        for attempt in range(max_retries):
            temp = base_temp + (0.1 * attempt)
            top_k = 50 + (10 * attempt)

            input_ids = self.tokeniser.encode(prompt_text, return_tensors='pt').to(self.device)

            if self.tokeniser.pad_token_id is None:
                gen_pad_token_id = self.tokeniser.eos_token_id
            else:
                gen_pad_token_id = self.tokeniser.pad_token_id

            max_len = len(input_ids[0]) + self.config['tokens_per_section']

            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=max_len,
                do_sample=True,
                temperature=temp,
                top_k=top_k,
                top_p=0.95,
                pad_token_id=gen_pad_token_id,
                repetition_penalty=self.config['repetition_penalty']
            )

            newly_generated_text = self.tokeniser.decode(output_sequences[0], skip_special_tokens=True).strip()
            newly_generated_text = newly_generated_text[len(prompt_text):].strip()

            if len(newly_generated_text.split()) > 2:
                return newly_generated_text

            logging.info(f"  > Generated text was too short. Retrying... (Attempt {attempt + 1}/{max_retries})")

        logging.warning("  > All retries failed. Using a random sentence from source text as a safety net.")
        try:
            with open(original_file_path, 'r', encoding='utf-8') as f:
                lines = [line for line in f.readlines() if line.strip()]
            return random.choice(lines).strip()
        except Exception:
            return "The world keeps turning on and on."

    def _format_section_into_lyrical_lines(self, text_block):
        words = text_block.replace('\n', ' ').split()
        if not words:
            return ""

        ideal_words_per_line = self.config['ideal_words_per_line']
        lyrical_lines = []
        current_line_words = []

        for word in words:
            current_line_words.append(word)
            if len(current_line_words) >= ideal_words_per_line:
                lyrical_lines.append(" ".join(current_line_words))
                current_line_words = []

        if current_line_words:
            lyrical_lines.append(" ".join(current_line_words))

        return "\n".join(lyrical_lines)

    def _load_model_for_generation(self, use_quantised=False):
        if self.model and self.tokeniser:
            logging.info("Model and tokeniser already loaded.")
            return

        if use_quantised and self.device.type == 'mps':
            logging.warning("\nWARNING: Quantisation is not supported on Apple Silicon (MPS).")
            logging.warning("Falling back to the full-precision model for generation.\n")
            use_quantised = False

        model_dir_to_load = self.config['quantised_output_dir'] if use_quantised else self.config['output_dir']
        logging.info(f"Loading model from: {model_dir_to_load}")

        if not os.path.exists(model_dir_to_load):
            logging.error(
                f"Model directory not found at '{model_dir_to_load}'. Please train or quantise the model first.")
            sys.exit(1)

        self.tokeniser = AutoTokenizer.from_pretrained(model_dir_to_load, trust_remote_code=True)
        if self.tokeniser.pad_token is None:
            self.tokeniser.pad_token = self.tokeniser.eos_token

        if use_quantised:
            logging.info("Loading QUANTISED model")
            model = AutoModelForCausalLM.from_pretrained(self.config['output_dir'], trust_remote_code=True)
            model.eval()
            torch.backends.quantized.engine = 'qnnpack'
            quantised_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            quantised_model.load_state_dict(torch.load(os.path.join(model_dir_to_load, "quantised_model.pt")))
            self.model = quantised_model
        else:
            logging.info(f"Loading FULL-PRECISION model")
            self.model = AutoModelForCausalLM.from_pretrained(model_dir_to_load, trust_remote_code=True)

        self.model.to(self.device)

    def _save_song(self, song_text, original_file_path, initial_idea):
        logging.info("Saving song to file...")
        try:
            name_without_ext = os.path.splitext(os.path.basename(original_file_path))[0]
            processed_name = re.sub(r'lyrics', '', name_without_ext, flags=re.IGNORECASE).strip().replace(' ', '_')

            song_subfolder = os.path.join(self.config['song_output_folder'], processed_name)
            os.makedirs(song_subfolder, exist_ok=True)

            model_name_file_safe = self.config['model_name'].replace("/", "_")

            i = 1
            while True:
                song_filename = f"{model_name_file_safe}_song_{i}.txt"
                full_path = os.path.join(song_subfolder, song_filename)
                if not os.path.exists(full_path):
                    break
                i += 1

            keys_to_save = ['model_name', 'training_epochs', 'batch_size', 'gradient_accumulation_steps', 'learning_rate', 'tokens_per_section', 'ideal_words_per_line', 'creativity_temperature', 'repetition_penalty', 'sanitise']
            config_string = "\n".join([f"{key}: {self.config[key]}" for key in keys_to_save if key in self.config])

            file_content = f"Generation Parameters\n{config_string}\n{'-' * 40}\nStarting song with initial idea: '{initial_idea}'\n{'-' * 40}\n\n{song_text}"

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_content)

            logging.info(f"Successfully saved song to: {full_path}")

        except Exception as e:
            logging.error(f"Could not save the song. {e}")

    def generate(self, file_path, use_random_prompt=False, use_quantised=False):
        logging.info("\n" + "=" * 50)
        logging.info("           LYRIC COMPOSITION ENGINE")
        logging.info("=" * 50 + "\n")

        try:
            self._load_model_for_generation(use_quantised)

            full_song_text, chorus_text = "", ""
            initial_idea = self._get_initial_prompt(file_path, use_random=use_random_prompt)
            logging.info(f"Starting song with initial idea: '{initial_idea}'")
            current_context = initial_idea

            for i, section_tag in enumerate(self.config['song_structure']):
                logging.info(f"Composing section {i + 1}/{len(self.config['song_structure'])}: {section_tag}...")

                prompt = f"{current_context}\n\n{section_tag}\n"

                if "chorus" in section_tag.lower() and chorus_text:
                    section_content = chorus_text
                else:
                    generated_block = self._generate_section(prompt, file_path)
                    section_content = self._format_section_into_lyrical_lines(generated_block)
                    if "chorus" in section_tag.lower():
                        chorus_text = section_content

                full_song_text += f"\n\n{section_tag}\n{section_content}"
                current_context = full_song_text

            final_song = full_song_text.strip()
            logging.info("\n" + "=" * 40 + " FINAL COMPOSITION " + "=" * 40)
            logging.info(f"Final Song:\n{final_song}")
            logging.info("=" * 100)

            if self.config['save_song_to_file']:
                self._save_song(final_song, file_path, initial_idea)

        except Exception as e:
            logging.error(f"An error occurred during the generation process: {e}", exc_info=True)
            sys.exit(1)

    def run_quantisation(self):
        if not os.path.exists(self.config['quantised_output_dir']):
            logging.info(f"Quantised model not found. Running quantisation script...")
            quantise_script_path = self.config.get('quantise_script_path', './extra/quantise_model.py')
            if not os.path.exists(quantise_script_path):
                logging.error(f"Quantisation script not found at '{quantise_script_path}'.")
                logging.error("Skipping quantisation and using full model.")
                return False
            else:
                command = [sys.executable, quantise_script_path, self.config['output_dir'], "-o",
                           os.path.dirname(self.config['quantised_output_dir'])]
                result = subprocess.run(command)
                if result.returncode != 0:
                    logging.error("Quantisation script failed to execute. Using full model.")
                    return False
            return True
        return True
