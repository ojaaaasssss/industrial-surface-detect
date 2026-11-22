"""
Helper script to download and organize Kaggle surface defect datasets
Usage:
  python scripts/setup_kaggle_dataset.py --dataset fantacher/neu-metal-surface-defects-data
  python scripts/setup_kaggle_dataset.py --dataset ravirajsinh45/real-life-industrial-dataset-of-casting-product
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

def download_and_prepare(dataset_name, output_dir='data/real'):
    output_path = Path(output_dir)
    def_dir = output_path / 'defective'
    non_def_dir = output_path / 'non_defective'
    def_dir.mkdir(parents=True, exist_ok=True)
    non_def_dir.mkdir(parents=True, exist_ok=True)

    temp_download = Path('data/temp_kaggle')
    temp_download.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name} using Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_name, path=str(temp_download), unzip=True)
        print("Download complete. Processing images into defective/ and non_defective/...")
    except Exception as e:
        print(f"Kaggle API error: {e}")
        print("Ensure 'kaggle.json' is placed at ~/.kaggle/kaggle.json (or C:\\Users\\<user>\\.kaggle\\kaggle.json)")
        return False

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    count_def = 0
    count_non_def = 0

    for file_path in temp_download.rglob('*'):
        if file_path.suffix.lower() in valid_exts:
            search_str = (file_path.name + ' ' + str(file_path.parent)).lower()
            # Identify normal / ok / non_defective surfaces
            if any(k in search_str for k in ['ok', 'good', 'normal', 'non_defective', 'clean', 'negative']):
                shutil.copy(file_path, non_def_dir / file_path.name)
                count_non_def += 1
            else:
                shutil.copy(file_path, def_dir / file_path.name)
                count_def += 1

    # Cleanup temp folder
    shutil.rmtree(temp_download, ignore_errors=True)
    print(f"Done! Copied {count_def} defective images to {def_dir}")
    print(f"Copied {count_non_def} non-defective images to {non_def_dir}")
    print(f"You can now train with: python train.py --data {output_dir} --epochs 15")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fantacher/neu-metal-surface-defects-data', help='Kaggle dataset slug')
    parser.add_argument('--output', default='data/real', help='Output folder')
    args = parser.parse_args()
    download_and_prepare(args.dataset, args.output)
