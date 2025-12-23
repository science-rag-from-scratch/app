from ntpath import exists
import os
import argparse
from pathlib import Path

import dotenv
from huggingface_hub import snapshot_download

dotenv.load_dotenv()

PROJECT_PATH = os.environ["PROJECT_PATH"]

EMB_MODEL_ID = "intfloat/multilingual-e5-large-instruct"
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

WEIGHTS_PATH = os.path.join(PROJECT_PATH, "models")

DEFAULT_EMB_PATH = os.path.join(PROJECT_PATH, "models/multilingual-e5-large-instruct")
DEFAULT_LLM_PATH = os.path.join(PROJECT_PATH, "models/Mistral-7B-Instruct-v0.3")


def download_model(repo_id: str, local_dir: str):
    print(f"Downloading {repo_id} to {local_dir}...")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ Successfully downloaded {repo_id} to {local_dir}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {repo_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument(
        "--emb-model-path",
        type=str,
        default=DEFAULT_EMB_PATH,
        help=f"Path to save embedding model (default: {DEFAULT_EMB_PATH})"
    )
    parser.add_argument(
        "--llm-model-path",
        type=str,
        default=DEFAULT_LLM_PATH,
        help=f"Path to save LLM model (default: {DEFAULT_LLM_PATH})"
    )
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.emb_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.llm_model_path).parent.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    success &= download_model(EMB_MODEL_ID, args.emb_model_path)
    success &= download_model(LLM_MODEL_ID, args.llm_model_path)
    
    if success:
        print("\n✓ All models downloaded successfully!")
    else:
        print("\n✗ Some models failed to download.")
        exit(1)


if __name__ == "__main__":
    main()

