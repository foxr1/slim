# main.py (for Cloud Run)
#
# A simplified Flask application designed to be run inside a Docker container on Cloud Run.
# This version uses a single, robust /generate endpoint.

import os
import random
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# --- CONFIGURATION ---
DEFAULT_CONFIG = {
    "song_structure": ["[Verse 1]", "[Chorus]", "[Verse 2]", "[Chorus]", "[Bridge]", "[Chorus]", "[Outro]"],
    "tokens_per_section": 75,
    "ideal_words_per_line": 10,
    "creativity_temperature": 0.7,
    "repetition_penalty": 1.25,
}

# --- GLOBAL CACHE & APP ---
models_cache = {}
app = Flask(__name__)
CORS(app)


# --- UTILITY FUNCTIONS ---
def get_model_and_tokenizer(model_name):
    if model_name in models_cache:
        return models_cache[model_name]
    model_path = os.path.join("./models_quantised", model_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path} inside the container.")

    print(f"Loading quantised model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    with torch.no_grad():
        model = AutoModelForCausalLM.from_config(config)
    model.eval()
    torch.backends.quantized.engine = 'qnnpack'
    quantised_model_shell = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    quantised_weights_path = os.path.join(model_path, "quantised_model.pt")
    if not os.path.exists(quantised_weights_path):
        raise FileNotFoundError(f"Quantised weights file 'quantised_model.pt' not found in {model_path}")
    state_dict = torch.load(quantised_weights_path, map_location=torch.device('cpu'))
    quantised_model_shell.load_state_dict(state_dict)
    model = quantised_model_shell

    device = torch.device("cpu")
    model.to(device)
    models_cache[model_name] = (model, tokenizer, device)
    return model, tokenizer, device


def generate_section(model, tokenizer, device, prompt_text, config):
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    max_len = len(input_ids[0]) + config['tokens_per_section']
    output_sequences = model.generate(
        input_ids=input_ids, max_length=max_len, do_sample=True,
        temperature=config['creativity_temperature'], top_k=50, top_p=0.95,
        pad_token_id=tokenizer.eos_token_id, repetition_penalty=config['repetition_penalty']
    )
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)[len(prompt_text):].strip()


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


# --- API ENDPOINTS ---
@app.route('/list_models', methods=['GET'])
def list_models():
    try:
        model_dir = "./models_quantised"
        if not os.path.isdir(model_dir):
            return jsonify({'error': f"Local directory '{model_dir}' not found."}), 404
        model_names = sorted([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))])
        return jsonify({'models': model_names})
    except Exception as e:
        return jsonify({'error': f"Could not list models: {e}"}), 500


@app.route('/random_prompt', methods=['POST'])
def get_random_prompt():
    try:
        data = request.get_json()
        model_name = data.get('model')
        if not model_name: return jsonify({'error': 'Model name not provided.'}), 400
        source_text_name = f"{model_name}.txt"
        local_path = os.path.join("./source_texts", source_text_name)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Source text file '{source_text_name}' not found.")
        with open(local_path, 'r', encoding='utf-8') as f:
            text = f.read()
        words = text.split()
        prompt = ' '.join(words[random.randint(0, len(words) - 5):][:5]) if len(words) > 5 else "The world is"
        return jsonify({'prompt': prompt})
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500


@app.route('/generate', methods=['POST'])
def handle_generation_request():
    try:
        data = request.get_json()
        if not data or 'model' not in data or 'prompt' not in data:
            return jsonify({'error': 'Invalid request. "model" and "prompt" keys are required.'}), 400

        model_name = data['model']
        prompt = data['prompt']
        generation_type = data.get('generation_type', 'song')  # Default to 'song'

        gen_config = DEFAULT_CONFIG.copy()
        gen_config['tokens_per_section'] = int(data.get('tokens_per_section', gen_config['tokens_per_section']))
        gen_config['ideal_words_per_line'] = int(data.get('ideal_words_per_line', gen_config['ideal_words_per_line']))
        gen_config['creativity_temperature'] = float(
            data.get('creativity_temperature', gen_config['creativity_temperature']))
        gen_config['repetition_penalty'] = float(data.get('repetition_penalty', gen_config['repetition_penalty']))

        model, tokenizer, device = get_model_and_tokenizer(model_name)

        song_structure = ["Verse 1"] if generation_type == 'verse' else gen_config['song_structure']

        full_song_text, chorus_text = "", ""
        current_context = prompt
        for section_tag in song_structure:
            gen_prompt = f"{current_context}\n\n{section_tag}\n"
            if "chorus" in section_tag.lower() and chorus_text:
                section_content = chorus_text
            else:
                generated_block = generate_section(model, tokenizer, device, gen_prompt, gen_config)
                section_content = format_section_into_lyrical_lines(generated_block, gen_config['ideal_words_per_line'])
                if "chorus" in section_tag.lower(): chorus_text = section_content
            full_song_text += f"\n\n{section_tag}\n{section_content}"
            current_context = full_song_text

        return jsonify({'lyrics': full_song_text.strip()})
    except FileNotFoundError as e:
        return jsonify({'error': f"Model '{data.get('model')}' not found. Please check the model name."}), 404
    except Exception as e:
        return jsonify({'error': f"An internal server error occurred: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
