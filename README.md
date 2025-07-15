# SLIM (Stylistic Lyric Inference Model)
A rudimentary program to fine-tune an LLM with the goal of generating lyrics based on an input text file.

## Installation
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

