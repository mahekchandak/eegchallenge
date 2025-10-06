#!/usr/bin/env python3
"""
Create NeurIPS EEG Challenge submission zip file
"""

import os
import shutil
import zipfile
from pathlib import Path


def create_submission_zip(output_path="my_submission.zip"):
    """
    Create submission zip file for NeurIPS EEG Challenge
    """
    
    # Files to include in submission
    submission_files = [
        "submission.py",
        "models/eegnet_pytorch.py", 
        "weights_challenge_1.pt",
        "weights_challenge_2.pt"  # Copy challenge_1 weights for both challenges
    ]
    
    # Create temporary directory for submission files
    temp_dir = Path("temp_submission")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Copy submission.py
        if os.path.exists("submission.py"):
            shutil.copy2("submission.py", temp_dir)
        else:
            raise FileNotFoundError("submission.py not found!")
        
        # Copy model file and rename to avoid import issues
        if os.path.exists("models/eegnet_pytorch.py"):
            shutil.copy2("models/eegnet_pytorch.py", temp_dir / "eegnet_pytorch.py")
        else:
            raise FileNotFoundError("models/eegnet_pytorch.py not found!")
        
        # Copy weights
        if os.path.exists("weights_challenge_1.pt"):
            shutil.copy2("weights_challenge_1.pt", temp_dir)
            # Use same weights for challenge 2 if challenge 2 weights don't exist
            if not os.path.exists("weights_challenge_2.pt"):
                shutil.copy2("weights_challenge_1.pt", temp_dir / "weights_challenge_2.pt")
            else:
                shutil.copy2("weights_challenge_2.pt", temp_dir)
        else:
            print("Warning: No trained weights found. Creating submission with random weights.")
        
        # Update submission.py to import from current directory
        with open(temp_dir / "submission.py", "r") as f:
            content = f.read()
        
        # Fix import path
        content = content.replace("from models.eegnet_pytorch import Model", 
                                "from eegnet_pytorch import Model")
        
        with open(temp_dir / "submission.py", "w") as f:
            f.write(content)
        
        # Create zip file
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    zipf.write(file_path, file_path.name)
        
        print(f"✅ Submission zip created: {output_path}")
        
        # List contents
        with zipfile.ZipFile(output_path, 'r') as zipf:
            print("\n📁 Contents:")
            for name in zipf.namelist():
                print(f"  - {name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating submission: {e}")
        return False
        
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def validate_submission(zip_path="my_submission.zip"):
    """
    Validate submission zip format
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            files = zipf.namelist()
            
            # Check required files
            required = ["submission.py"]
            missing = [f for f in required if f not in files]
            
            if missing:
                print(f"Missing required files: {missing}")
                return False
            
            # Check no folders
            folders = [f for f in files if f.endswith('/')]
            if folders:
                print(f"Submission contains folders (not allowed): {folders}")
                return False
            
            print("Submission format validation passed")
            return True
            
    except Exception as e:
        print(f"Error validating submission: {e}")
        return False


if __name__ == "__main__":
    print("Creating NeurIPS EEG Challenge submission...")
    
    success = create_submission_zip()
    if success:
        validate_submission()
        print("\n🚀 Ready to submit! Upload my_submission.zip to the challenge platform.")
    else:
        print("\n❌ Submission creation failed!")
