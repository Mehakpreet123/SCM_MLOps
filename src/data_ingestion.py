import os
import argparse
import zipfile
import subprocess

def download_from_kaggle(dataset: str, output_dir: str):
    """
    Downloads a dataset from Kaggle using the Kaggle API.
    Example dataset: 'kaggle dataset download -d <dataset>'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📦 Downloading dataset '{dataset}' from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", output_dir], check=True)

    # Unzip downloaded file
    for file in os.listdir(output_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(output_dir, file)
            print(f"📂 Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)
            print(f"✅ Extracted to {output_dir}")

def main(dataset: str, output_dir: str):
    download_from_kaggle(dataset, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Kaggle")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Kaggle dataset identifier, e.g., 'mkechinov/ecommerce-behavior-data-from-multi-category-store'"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/raw", 
        help="Directory to store downloaded data"
    )
    args = parser.parse_args()
    main(args.dataset, args.output_dir)
