import os
import requests
import zipfile
from pathlib import Path
import shutil

# URL for a small subset of COCO val2017 images (hosted by Ultralytics or similar, or we can download individual images)
# For simplicity and reliability, we will download the 'coco8.zip' from Ultralytics which is very small, 
# or we can try to download a few specific images if we want more control.
# Ultralytics auto-downloads datasets, but we want to control it for calibration.
# Let's use the 'ultralytics' library to download the 'coco8' dataset which is tiny (8 images), 
# or 'coco128' (128 images) which is better for calibration.

from ultralytics.utils.downloads import download


def download_coco128():
    """Downloads the COCO128 dataset."""
    url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'
    download_dir = Path('datasets')
    download_dir.mkdir(exist_ok=True)
    
    zip_path = download_dir / 'coco128.zip'
    
    if not (download_dir / 'coco128').exists():
        print("Downloading COCO128 dataset...")
        download(url, dir=download_dir, unzip=True, delete=False)
        print("Download complete.")
    else:
        print("COCO128 dataset already exists.")

    return download_dir / 'coco128'

if __name__ == "__main__":
    data_path = download_coco128()
    print(f"Data is ready at: {data_path.absolute()}")
