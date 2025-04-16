import sys

MIN_PYTHON_VERSION = (3, 12)

if sys.version_info < MIN_PYTHON_VERSION:
    print(f"❌ At least Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} needed.")
    sys.exit(1)

import shutil

if shutil.which("pip") is None:
    print("❌ pip module was not found, please install it.")

import subprocess
from pathlib import Path

ENVIRONMENT_BASE_DIR = Path("../../")
ENVIRONMENT_NAME = Path("auto-os")
ENVIRONMENT = ENVIRONMENT_BASE_DIR / ENVIRONMENT_NAME

DATASETS_BASE_LOCATION = ENVIRONMENT_BASE_DIR / Path("datasets")

CULANE_DATASET_LOCATION = DATASETS_BASE_LOCATION / Path("CuLane/")
CULANE_DATASET_DATA = {
    "test": ["https://drive.google.com/file/d/1Z6a463FQ3pfP54HMwF3QS5h9p2Ch3An7",
             "https://drive.google.com/file/d/1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8",
             "https://drive.google.com/file/d/1daWl7XVzH06GwcZtF4WD8Xpvci5SZiUV"],
    "train-validation": [
             "https://drive.google.com/file/d/14Gi1AXbgkqvSysuoLyq1CsjFSypvoLVL",
             "https://drive.google.com/file/d/1AQjQZwOAkeBTSG_1I9fYn8KBcxBBbYyk",
             "https://drive.google.com/file/d/1PH7UdmtZOK3Qi3SBqtYOkWSH2dpbfmkL"]
}
CULANE_ARCHIVE_MAP = {
    f"{CULANE_DATASET_DATA["test"][0]}": "driver_37_30frame",
    f"{CULANE_DATASET_DATA["test"][1]}": "driver_100_30frame",
    f"{CULANE_DATASET_DATA["test"][2]}": "driver_193_90frame",
    f"{CULANE_DATASET_DATA["train-validation"][0]}": "driver_23_30frame",
    f"{CULANE_DATASET_DATA["train-validation"][1]}": "driver_161_90frame",
    f"{CULANE_DATASET_DATA["train-validation"][2]}": "driver_182_30frame"
}
CULANE_ARCHIVE_EXTENSION = ".tar.gz"

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
    import gdown

    Path(DATASETS_BASE_LOCATION).mkdir(parents=True, exist_ok=True)
    Path(CULANE_DATASET_LOCATION).mkdir(parents=True, exist_ok=True)

    test_dir = CULANE_DATASET_LOCATION / "test"
    train_val_dir = CULANE_DATASET_LOCATION / "train-validation"

    test_dir.mkdir(parents=True, exist_ok=True)
    train_val_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Retrieving CuLane dataset to {CULANE_DATASET_LOCATION}")

    for url in CULANE_DATASET_DATA["test"]:
        if not Path(test_dir / f"{CULANE_ARCHIVE_MAP[url]}{CULANE_ARCHIVE_EXTENSION}").exists():
            file_id = url.split("/d/")[1]
            output = test_dir / f"{CULANE_ARCHIVE_MAP[url]}{CULANE_ARCHIVE_EXTENSION}"
            gdown.download(id=file_id, output=str(output), quiet=False)

    for url in CULANE_DATASET_DATA["train-validation"]:
        if not Path(train_val_dir / f"{CULANE_ARCHIVE_MAP[url]}{CULANE_ARCHIVE_EXTENSION}").exists():
            file_id = url.split("/d/")[1]
            output = train_val_dir / f"{CULANE_ARCHIVE_MAP[url]}{CULANE_ARCHIVE_EXTENSION}"
            gdown.download(id=file_id, output=str(output), quiet=False)
    
    print(f"📥 Unzipping archives")

    for url in CULANE_DATASET_DATA["test"]:
        if not Path(test_dir / f"{CULANE_ARCHIVE_MAP[url]}").exists() :
            shutil.unpack_archive(test_dir / f"{CULANE_ARCHIVE_MAP[url]}{CULANE_ARCHIVE_EXTENSION}", test_dir)
    
    for url in CULANE_DATASET_DATA["train-validation"]:
        if not Path(train_val_dir / f"{CULANE_ARCHIVE_MAP[url]}").exists() :
            shutil.unpack_archive(train_val_dir / f"{CULANE_ARCHIVE_MAP[url]}{CULANE_ARCHIVE_EXTENSION}", train_val_dir)

def main():
    venv_setup()

    install_requirements()

    dataset_setup()

main()