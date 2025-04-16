import sys

MIN_PYTHON_VERSION = (3, 12)

if sys.version_info < MIN_PYTHON_VERSION:
    print(f"❌ At least Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} needed.")
    sys.exit(1)

import shutil

if shutil.which("pip") is None:
    print("❌ pip module was not found, please install it.")

import subprocess
import gdown
from pathlib import Path

ENVIRONMENT_BASE_DIR = Path("../../")
ENVIRONMENT_NAME = Path("auto-os")
ENVIRONMENT = ENVIRONMENT_BASE_DIR / ENVIRONMENT_NAME

DATASETS_BASE_LOCATION = ENVIRONMENT_BASE_DIR / Path("../../datasets")

CULANE_DATASET_LOCATION = DATASETS_BASE_LOCATION / Path("/CuLane/")
CULANE_DATASET_DATA = {
    "test": ("https://drive.google.com/open?id=1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7&authuser=0",
             "https://drive.google.com/open?id=1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8&authuser=0",
             "https://drive.google.com/open?id=1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV&authuser=0"),
    "train-validation": (
             "https://drive.google.com/open?id=14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL&authuser=0",
             "https://drive.google.com/open?id=1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk&authuser=0",
             "https://drive.google.com/open?id=1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL&authuser=0")
}

def venv_setup():
    if ENVIRONMENT.exists():
        print("📦 Virtual environment already exists.")
    else:
        print("📦 Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(ENVIRONMENT)])
        print("✅ Environment has been successfully created.")

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies are installed.")
    except subprocess.CalledProcessError:
        print("❌ Error during installation of requirements.txt")

def dataset_setup():
    Path(DATASETS_BASE_LOCATION).mkdir(parents=True, exist_ok=True)

    Path(CULANE_DATASET_LOCATION).mkdir(parents=True, exist_ok=True)

    for sample in CULANE_DATASET_DATA["test"]:
        gdown.download(sample, ".")


def main():
    venv_setup()

    install_requirements()

main()