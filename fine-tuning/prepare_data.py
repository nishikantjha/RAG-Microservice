#!/usr/bin/env python3
"""
prepare_data.py
──────────────────────────────────────────────────────────────────────────────
Prepares and validates the training data for fine-tuning.

WHAT THIS DOES:
  1. Loads the raw JSONL training data
  2. Validates each entry has required fields
  3. Formats each entry into the "Alpaca" instruction format
     (the standard format for instruction fine-tuning)
  4. Splits into train/validation sets (90%/10%)
  5. Saves formatted data ready for the trainer

WHY Alpaca format?
  Llama and most open-source models were further instruction-tuned using
  the Alpaca format, so they understand this structure natively:
  
  ### Instruction:
  {task description}
  
  ### Input:
  {optional context/input}
  
  ### Response:
  {expected output}

WHY 90/10 train/val split?
  Standard ML practice. The validation set lets you monitor whether the
  model is learning (decreasing val loss) or just memorizing training data
  (overfitting — low train loss, high val loss). For a 8-example dataset
  this split is just for demonstration; in practice you need 100+ examples.
──────────────────────────────────────────────────────────────────────────────
"""

import json
import random
from pathlib import Path


def format_alpaca_prompt(instruction: str, input_text: str, output: str = "") -> str:
    """
    Format a training example into the Alpaca instruction template.
    
    The model learns to complete the pattern: given ### Instruction + ### Input,
    produce the ### Response. At inference time, we stop at ### Response:
    and let the model generate the rest.
    """
    if input_text.strip():
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        # No input — just instruction and response
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return prompt


def load_and_validate(data_path: Path) -> list[dict]:
    """Load JSONL and validate each entry has required fields."""
    entries = []
    required_fields = {"instruction", "output"}
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ERROR: Line {line_num} is invalid JSON: {e}")
                continue
            
            missing = required_fields - set(entry.keys())
            if missing:
                print(f"  WARNING: Line {line_num} missing fields: {missing}")
                continue
            
            entries.append(entry)
    
    return entries


def main():
    print("═══ Fine-Tuning Data Preparation ═══")
    
    base_dir = Path(__file__).parent
    raw_data_path = base_dir / "data" / "training_data.jsonl"
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(exist_ok=True)
    
    # ── Step 1: Load and validate ──────────────────────────────────────────────
    print(f"\n[1/4] Loading data from: {raw_data_path}")
    entries = load_and_validate(raw_data_path)
    print(f"      Valid entries loaded: {len(entries)}")
    
    # ── Step 2: Format into Alpaca prompt structure ────────────────────────────
    print("\n[2/4] Formatting into Alpaca instruction template...")
    formatted = []
    for entry in entries:
        formatted_text = format_alpaca_prompt(
            instruction=entry["instruction"],
            input_text=entry.get("input", ""),
            output=entry["output"],
        )
        formatted.append({
            "text": formatted_text,
            "instruction": entry["instruction"],
            "input": entry.get("input", ""),
            "output": entry["output"],
        })
    
    # Show an example
    print("\n  Example formatted entry:")
    print("  " + "─" * 60)
    example_lines = formatted[0]["text"].split("\n")
    for line in example_lines[:8]:
        print(f"  {line}")
    print("  ...")
    print("  " + "─" * 60)
    
    # ── Step 3: Split train/validation ────────────────────────────────────────
    print(f"\n[3/4] Splitting into train/validation (90%/10%)...")
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(formatted)
    
    split_idx = max(1, int(len(formatted) * 0.9))
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]
    
    print(f"      Train examples : {len(train_data)}")
    print(f"      Val examples   : {len(val_data)}")
    
    # ── Step 4: Save processed data ────────────────────────────────────────────
    print("\n[4/4] Saving processed data...")
    
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"      Saved train data to: {train_path}")
    print(f"      Saved val data to  : {val_path}")
    
    print("\n═══ Data preparation complete ═══")
    print("\nNext step: Run the training script:")
    print("  python train_qlora.py")


if __name__ == "__main__":
    main()
