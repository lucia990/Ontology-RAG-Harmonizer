import os
from transformers import AutoModel, AutoTokenizer
import argparse

def download_and_save_model(model_name: str, output_dir: str):
    """Downloads a Hugging Face model and tokenizer and saves them."""
    print(f"Downloading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save tokenizer and model
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a Hugging Face model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Hugging Face model (e.g., 'sentence-transformers/all-MiniLM-L6-v2').")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the model and tokenizer files.")
    args = parser.parse_args()

    download_and_save_model(args.model_name, args.output_dir)