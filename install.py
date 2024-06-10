import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    "torch",
    "opencv-python",
    "numpy",
    "huggingface_hub",
    "torchvision"
]

for package in packages:
    install(package)

print("All packages installed successfully.")
