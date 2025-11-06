# SLIM (Stylistic Lyric Inference Model)
SLIM is an automated pipeline designed to fine-tune a pre-trained transformer model on a given text corpus and subsequently generate a new body of text, in the style of lyrics.

## Methodology
### Tested Models
Models were retrieved from [HuggingFace](https://huggingface.co/), and the following models have been tested and are supported:
- [GPT-2 Medium](https://huggingface.co/openai-community/gpt2-medium)
- [DistilGPT2](https://huggingface.co/distilbert/distilgpt2)
- [OPT](https://huggingface.co/facebook/opt-350m)
- [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

### Pre-Processing
The input corpus is first sanitised to remove noise and artifacts that might affect the training process. For example, this might include textual noise from academic or annotated texts, such as citation markers. Normalisation is also applied, with abbreviations like `"i.e."` being replaced with `"that is"`, and whitespace reduced to single spaces. This text is then segmented into individual sentences and tokenised.

Strictly **decoder models** are used in this project, as experiments with encoder models proved invalid for the specific task. These models are trained using Casual Language Modelling (CLM), where the cleaned sentences are presented as a continuous stream of text.

### Fine-Tuning
Here, we adapt the pre-trained model to the specific linguistic domain of the input corpus, this is done through "fine-tuning", where the pre-trained weights of the model are updated through backpropagation, where the model's parameters are adjusted to better fit the training data.

### Text Generation
A prompt is required for generating the "lyrics", which can either be specified by the user, or randomly selected from the corpus. If the prompt is specified, this prompt is used to find a sentence from the corpus that best matches the prompt. 

The pipeline then iterated through a defined `song_structure`, defined in the [config](config.yaml). For each section, a new prompt is constructed, this is the entire song generated so far, plus the new section tag.

Through autoregressive generation, the model generates the text token-by-token, with a restriction coming from the `tokens_per_section` parameter. SLIM also employs a "creative retry", whereby if the model fails to produce meaningful text (the output is too short), the function automatically retries up to 3 times, each time increasing the `creativity_temperature`, to encourage the model to explore less probable, more novel word choices. If all retries fail, the system abandons generation for that section and samples a random, clean sentence from the corpus.

To mimic song structures, the text generated for the first `[Chorus]` section is cached, and all subsequent `[Chorus]` sections are populated with this cached text.

The final block of text for each section is then passed to a heuristic formatter, which splits the text into shorted lines based on the `ideal_words_per_line` parameter. The output is then saved to a `.txt` file, with the generation parameters to ensure traceability and reproducibility.

## Installation
If using CUDA, navigate to https://developer.nvidia.com/cuda-downloads, and install based on the provided instructions.

Recommended using Python 3.10, install all requirements with:
```bash
pip install .
```

## Usage
Use a `.txt` file as the input, after installing, use:
```bash
generate-lyrics input_file.txt --parameters
```
Some example input text can be found in the `input_files` directory.

### Parameters
Either adjust the program with `config.yaml`, or use the following for help on commands: 
```bash
generate-lyrics --h
```

***DISCLAIMER**:  Generated content may be unpredictable.*

