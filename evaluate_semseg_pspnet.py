"""Evaluate semseg PSPNet on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.semseg.models import make_build

if __name__ == "__main__":
    main("evaluate", task="semseg", version="pspnet", build_model=make_build("pspnet"))
