# Message Extractor
# This script is built to extract all text from messaged downloaded from [Facebook Messenger](https://www.facebook.com/help/messenger-app/677912386869109/?helpref=related_articles), as an experiment to be used as training data.

import json
import os


def concatenate_message_content(input_file_path, output_file_path):
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at '{input_file_path}'")
        return

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file_path}'. Make sure it is a valid JSON file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    # Check if the top-level 'messages' key exists and is a list. (As per output from Facebook Messenger data)
    if 'messages' not in data or not isinstance(data['messages'], list):
        print("Error: JSON file must contain a 'messages' key with a list of message objects.")
        return

    content_list = [
        message['text']
        for message in data['messages']
        if 'text' in message and isinstance(message['text'], str)
    ]

    concatenated_text = "\n".join(content_list)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(concatenated_text)
        print(f"Successfully concatenated content to '{output_file_path}'")
    except Exception as e:
        print(f"An error occurred while writing to the output file: {e}")


if __name__ == '__main__':
    input_filename = "example.json"
    output_filename = "concatenated_messages.txt"
    concatenate_message_content(input_filename, output_filename)
