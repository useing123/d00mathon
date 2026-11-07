"""Fine-tuning toolkit (dataset builder, augmentation, QLoRA trainer helpers).

Nothing here is imported by the evaluator; heavy ML deps (torch/transformers/peft/trl) are imported lazily
inside scripts/finetune_lora.py so the core package and tests run without them. See docs/FINETUNE.md.
"""
