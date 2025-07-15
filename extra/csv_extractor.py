# CSV Extractor
# A utility script that reads a CSV file, extracts all text from a
# specified column, and concatenates it into a single output text file.

import csv
import argparse
import os
import sys


def extract_and_concatenate_csv(input_file_path, output_file_path, column_header):
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at '{input_file_path}'")
        sys.exit(1)

    content_list = []

    try:
        with open(input_file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)

            # Check if the specified column header exists in the file
            if column_header not in reader.fieldnames:
                print(f"Error: Column header '{column_header}' not found in '{input_file_path}'.")
                print(f"Available headers are: {', '.join(reader.fieldnames)}")
                sys.exit(1)

            for row in reader:
                cell_content = row.get(column_header)
                if cell_content and isinstance(cell_content, str):
                    content_list.append(cell_content.strip())

    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV file: {e}")
        sys.exit(1)

    # Join the list of content strings, with a newline character after each one.
    concatenated_text = "\n".join(content_list)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(concatenated_text)
        print(
            f"Successfully extracted and concatenated {len(content_list)} rows from column '{column_header}' to '{output_file_path}'")
    except Exception as e:
        print(f"An error occurred while writing to the output file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extracts and concatenates text from a specified column in a CSV file."
    )

    parser.add_argument(
        "input_file",
        metavar="INPUT_CSV_FILE",
        type=str,
        help="The path to the input .csv file."
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        type=str,
        required=True,
        help="The path for the final output .txt file."
    )

    parser.add_argument(
        "-c", "--column",
        dest="column_header",
        type=str,
        required=True,
        help="The header of the column to extract text from."
    )

    args = parser.parse_args()

    extract_and_concatenate_csv(args.input_file, args.output_file, args.column_header)


if __name__ == "__main__":
    main()
