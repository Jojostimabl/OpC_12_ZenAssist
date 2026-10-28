"""Les fichiers temporaires des tests restent dans le workspace local."""
import os
from pathlib import Path


def pytest_configure():
    target = Path(__file__).resolve().parents[1] / ".test-temp"
    target.mkdir(exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(target)
