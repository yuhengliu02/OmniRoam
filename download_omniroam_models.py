#!/usr/bin/env python3
'''
ADOBE CONFIDENTIAL
Copyright 2026 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
'''

from huggingface_hub import snapshot_download


def download_models():
    """Download all OmniRoam model checkpoints."""
    
    print("="*60)
    print("Downloading OmniRoam Model Checkpoints")
    print("="*60)
    print("Repository: yuhengliu02/OmniRoam\n")
    
    try:
        # Download entire repository
        local_dir = snapshot_download(
            repo_id="yuhengliu02/OmniRoam",
            repo_type="model",
            local_dir="models/OmniRoam",
            local_dir_use_symlinks=False
        )
        
        print(f"✓ All models downloaded successfully!")
        print(f"  Location: {local_dir}\n")
        print("Model structure:")
        print("  models/OmniRoam/")
        print("  ├── Preview/")
        print("  ├── Self-forcing/")
        print("  └── Refine/")
        
    except Exception as e:
        print(f"✗ Failed to download models: {e}\n")
        print("Please try manual download:")
        print("  1. Visit: https://huggingface.co/yuhengliu02/OmniRoam")
        print("  2. Download the Preview/, Self-forcing/, and Refine/ folders")
        print("  3. Place them in: models/OmniRoam/")
        return False
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = download_models()
    exit(0 if success else 1)

