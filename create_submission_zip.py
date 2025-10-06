#!/usr/bin/env python3
"""
Create submission zip file for NeurIPS EEG Challenge
"""

import os
import zipfile

def create_submission_zip(output_path="my_submission.zip"):
    """
    Create submission zip file with no folders - single level depth only
    """
    # Files to include
    files = [
        "submission.py",
        "weights_challenge_1.pt",
        "weights_challenge_2.pt"
    ]
    
    # Create zip file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            if os.path.exists(file):
                zipf.write(file, os.path.basename(file))
            else:
                print(f"Warning: {file} not found")

if __name__ == "__main__":
    create_submission_zip()