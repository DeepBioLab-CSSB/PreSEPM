#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: ESM-2 Embedding Extraction for Protein Sequences
Description:
    This script extracts per-sequence embeddings from protein amino acid sequences
    using Facebook FAIR's ESM-2 pretrained models. It supports CPU/GPU execution,
    configurable model selection, and flexible input/output file paths.
"""

import esm
import torch
import pandas as pd
import csv
import argparse
import os
import traceback


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract protein embeddings using ESM-2")
    parser.add_argument(
        "--model_name",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="ESM-2 model name, e.g., esm2_t33_650M_UR50D"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input CSV file containing sequences (two columns: ID, sequence)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output CSV file for embeddings"
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Maximum number of lines to process (default: all)"
    )
    return parser.parse_args()


def read_specific_line(csv_file, line_number):
    """
    Read a specific line from a CSV file (0-based index).
    Expected format: ID, sequence
    """
    df_row = pd.read_csv(csv_file, header=None, skiprows=line_number, nrows=1)
    seq_list = df_row.iloc[0].tolist()
    elements = [(seq_list[0], seq_list[1])]
    return elements


def array_write_to_csvfile(array, outfile, line_number):
    """
    Append a sequence embedding array to a CSV file.
    """
    array_np = [t.cpu().numpy() for t in array]
    df_out = pd.DataFrame(array_np)
    df_out.to_csv(outfile, mode='a', header=False, index=False)
    print(f"Processed line {line_number}")


def esm2_extract_embeddings(model, alphabet, batch_converter, data, device):
    """
    Extract per-sequence embeddings using ESM-2.
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations = [
        token_representations[i, 1:tokens_len - 1].mean(0)
        for i, tokens_len in enumerate(batch_lens)
    ]
    return sequence_representations


def main():
    args = parse_args()

    # Detect device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # Load ESM-2 model
    print(f"Loading model: {args.model_name} ...")
    model, alphabet = esm.pretrained.__dict__[args.model_name]()
    batch_converter = alphabet.get_batch_converter()
    model.eval().to(device)

    # Remove duplicate lib warning
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    max_lines = args.max_lines
    i = 0
    while True:
        try:
            if max_lines is not None and i >= max_lines:
                break
            data = read_specific_line(args.input_file, i)
            embeddings = esm2_extract_embeddings(model, alphabet, batch_converter, data, device)
            array_write_to_csvfile(embeddings, args.output_file, i)
            i += 1
        except pd.errors.EmptyDataError:
            break  # End of file
        except Exception as e:
            print(f"❌ Error processing line {i}: {e}")
            traceback.print_exc()
            # Write empty row for failed sequence
            with open(args.output_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([])
            i += 1
            continue

    print("✅ Processing completed.")


if __name__ == "__main__":
    main()
