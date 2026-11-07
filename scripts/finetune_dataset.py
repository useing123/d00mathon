#!/usr/bin/env python
"""Build the CQA SFT dataset: python scripts/finetune_dataset.py [--augment] [--no-distill] ... (see cqa/finetune/dataset.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.finetune.dataset import main  # noqa: E402

if __name__ == "__main__":
    main()
