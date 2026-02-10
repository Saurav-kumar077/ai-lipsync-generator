#!/usr/bin/env python3
"""
Model Download Script for AI Lip-Sync Generator

This script downloads the necessary Wav2Lip model checkpoints
and sets up the required files for lip-sync generation.
"""

import os
import sys
import urllib.request
import hashlib
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download a file with progress bar"""
    print(f"Downloading {os.path.basename(output_path)}...")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                miniters=1, desc=os.path.basename(output_path)) as t:
            urllib.request.urlretrieve(url, filename=output_path,
                                     reporthook=t.update_to)
        print(f"✓ Downloaded successfully: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {url}: {str(e)}")
        return False


def verify_file(filepath, expected_size_mb=None):
    """Verify file exists and optionally check size"""
    if not os.path.exists(filepath):
        return False
    
    if expected_size_mb:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb < expected_size_mb * 0.9:  # Allow 10% variance
            print(f"Warning: File size ({size_mb:.1f}MB) seems smaller than expected ({expected_size_mb}MB)")
            return False
    
    return True


def setup_wav2lip_models():
    """Download Wav2Lip model checkpoints"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Wav2Lip model URLs
    models = {
        "wav2lip.pth": {
            "url": "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA",
            "size_mb": 148
        },
        "wav2lip_gan.pth": {
            "url": "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EhJIQRBkEHtCswuPFhtJLPYBBYB0RPcaI4b7u_3G7vXFfA",
            "size_mb": 148
        }
    }
    
    print("=" * 60)
    print("Wav2Lip Model Download")
    print("=" * 60)
    print("\nThis script will download Wav2Lip model checkpoints.")
    print("Note: Models are large (~148MB each). This may take a while.\n")
    
    # Ask which model to download
    print("Available models:")
    print("1. wav2lip.pth (Standard model - recommended for CPU)")
    print("2. wav2lip_gan.pth (GAN model - higher quality, slower)")
    print("3. Both models")
    
    choice = input("\nEnter your choice (1/2/3) [default: 1]: ").strip() or "1"
    
    models_to_download = []
    if choice == "1":
        models_to_download = ["wav2lip.pth"]
    elif choice == "2":
        models_to_download = ["wav2lip_gan.pth"]
    elif choice == "3":
        models_to_download = ["wav2lip.pth", "wav2lip_gan.pth"]
    else:
        print("Invalid choice. Downloading standard model.")
        models_to_download = ["wav2lip.pth"]
    
    print("\n" + "=" * 60)
    
    success_count = 0
    for model_name in models_to_download:
        model_path = os.path.join(models_dir, model_name)
        model_info = models[model_name]
        
        # Check if already exists
        if verify_file(model_path, model_info["size_mb"]):
            print(f"✓ {model_name} already exists and verified")
            success_count += 1
            continue
        
        # Download
        if download_file(model_info["url"], model_path):
            if verify_file(model_path):
                success_count += 1
            else:
                print(f"✗ Downloaded file verification failed for {model_name}")
    
    print("\n" + "=" * 60)
    print(f"Download Summary: {success_count}/{len(models_to_download)} models ready")
    print("=" * 60)
    
    if success_count == len(models_to_download):
        print("\n✓ All models downloaded successfully!")
        print("\nYou can now run the lip-sync generator:")
        print("  python generate_lipsync.py --text 'Your text' --image inputs/images/person.jpg")
        return True
    else:
        print("\n✗ Some models failed to download.")
        print("\nAlternative download method:")
        print("1. Visit: https://github.com/Rudrabha/Wav2Lip")
        print("2. Download models manually from the 'Pretrained Models' section")
        print(f"3. Place them in the '{models_dir}/' directory")
        return False


def main():
    """Main function"""
    print("\nAI Lip-Sync Generator - Model Setup\n")
    
    # Check Python version
    if sys.version_info < (3, 7) or sys.version_info >= (3, 10):
        print("Warning: This project is tested with Python 3.7-3.9")
        print(f"You are using Python {sys.version_info.major}.{sys.version_info.minor}")
        print("Continuing anyway...\n")
    
    # Setup models
    success = setup_wav2lip_models()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
