import os
import subprocess
import shutil
import zipfile

def run_training():
    python_path = "python3"  # or full path to your venv Python
    print("Training EEGNet model for Challenge 1...")
    subprocess.run([python_path, "scripts/train_pytorch_eegnet.py", 
                   "--save_weights", "weights_challenge_1.pt",
                   "--tmax", "2.0",
                   "--epochs", "10"], check=True)  # Train for 10 epochs
    print("Training EEGNet model for Challenge 2...")
    subprocess.run([python_path, "scripts/train_pytorch_eegnet.py", 
                   "--save_weights", "weights_challenge_2.pt",
                   "--tmax", "2.0",
                   "--epochs", "10"], check=True)

def check_weights():
    files = ["weights_challenge_1.pt", "weights_challenge_2.pt"]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found after training!")

def create_zip():
    print("Creating my_submission.zip in the correct format...")
    files = ["submission.py", "weights_challenge_1.pt", "weights_challenge_2.pt"]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file {f} not found!")
    zip_path = os.path.abspath("../my_submission.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in files:
            z.write(f, arcname=os.path.basename(f))
    print(f"Created {zip_path}")

if __name__ == "__main__":
    #run_training()
    #check_weights()
    create_zip()