#!/usr/bin/env python3
"""
train_qlora.py
──────────────────────────────────────────────────────────────────────────────
QLoRA fine-tuning script for Llama-3.2-3B on the TISIX publishing dataset.

──────────────────────────────────────────────────────────────────────────────
WHY QLoRA and not full fine-tuning?
  Full fine-tuning: Update ALL 3 billion parameters. Needs 24GB+ VRAM. ❌
  LoRA:  Freeze original model. Add small adapter matrices. Train only ~0.1%.
         Still needs ~10GB VRAM because base model loaded in float16.
  QLoRA: Same as LoRA but load the BASE model in 4-bit (not float16).
         Only needs ~4-6GB VRAM. Runs on a MacBook Pro or a single consumer GPU. ✅
  
  The quality loss from 4-bit quantization is minimal (2-3%) but the VRAM
  savings are massive (4x reduction). For 3B parameter models, QLoRA is the
  standard approach in 2025.

WHY Llama-3.2-3B (not 7B, not 70B)?
  - 3B: ~6GB disk, 4-6GB VRAM with QLoRA. Trainable on MacBook / single GPU.
  - 7B: ~14GB disk, 8-12GB VRAM. Needs a decent GPU.
  - 70B: ~140GB disk, multi-GPU required.
  → 3B gives a good balance of capability and trainability for a demo.

WHY these LoRA hyperparameters?
  r=16: The "rank" of the LoRA adapter matrices. Higher rank = more parameters
        = more capacity to learn new behavior = slower training. 16 is standard.
  lora_alpha=32: Scaling factor. Rule of thumb: 2x the rank.
  target_modules: Which layers to add adapters to. q_proj and v_proj are the
        attention Query and Value projections — the most impactful layers.
  lora_dropout=0.05: Small regularization to prevent overfitting. Low because
        our dataset is small.

MODEL NOTE:
  Llama-3.2-3B requires a Hugging Face account and accepting Meta's license at:
  https://huggingface.co/meta-llama/Llama-3.2-3B
  
  After accepting, run: huggingface-cli login
  Then the model will download automatically (~6GB).

  ALTERNATIVE (no login needed): Use "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  It's smaller (1.1B params) and freely available, good for testing the pipeline.
──────────────────────────────────────────────────────────────────────────────
"""

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer


# ── Configuration ──────────────────────────────────────────────────────────────

# Model to fine-tune
# Change to "TinyLlama/TinyLlama-1.1B-Chat-v1.0" for a no-login alternative
BASE_MODEL = "meta-llama/Llama-3.2-3B"

# Where to save the fine-tuned adapter
OUTPUT_DIR = Path(__file__).parent / "output" / "tisix-publisher-adapter"

# Training data (prepared by prepare_data.py)
TRAIN_DATA = Path(__file__).parent / "data" / "processed" / "train.jsonl"
VAL_DATA = Path(__file__).parent / "data" / "processed" / "val.jsonl"

# ── QLoRA: 4-bit quantization config ──────────────────────────────────────────
# This is what makes QLoRA different from regular LoRA
# It loads the BASE model in 4-bit precision, saving ~4x VRAM

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,           # Load base model weights in 4-bit (not 16-bit)
    bnb_4bit_use_double_quant=True,  # Quantize the quantization constants too (saves a bit more)
    bnb_4bit_quant_type="nf4",   # NF4 = Normal Float 4 — best for normally distributed weights
    bnb_4bit_compute_dtype=torch.float16,  # Run computations in fp16, store in 4bit
)

# ── LoRA adapter config ────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=16,                        # Rank — size of the adapter matrices
    lora_alpha=32,               # Scaling: effective learning rate = (alpha/r) * lr
    # These are the attention layers we're adapting
    # q=query, k=key, v=value, o=output projections in the attention mechanism
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",                 # Don't adapt bias terms (not worth the compute)
    task_type=TaskType.CAUSAL_LM,  # Language modelling task
)


def main():
    print("═══ TISIX QLoRA Fine-Tuning ═══")
    print(f"Base model : {BASE_MODEL}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Train data : {TRAIN_DATA}")
    print()

    # Check GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon MPS available — using Metal GPU")
        # NOTE: bitsandbytes 4-bit quantization is NOT supported on MPS.
        # If you're on a MacBook, you need to disable QLoRA and use regular LoRA
        # or train in fp16 without quantization. The training will be slow.
        print("WARNING: 4-bit quantization not supported on MPS.")
        print("         Training will use fp16 without quantization.")
        print("         For true QLoRA, use a CUDA GPU (Linux + Nvidia).")
    else:
        device = "cpu"
        print("No GPU detected — training on CPU (will be VERY slow, demo only)")

    print()

    # ── Step 1: Load tokenizer ─────────────────────────────────────────────────
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # LLMs don't have a default padding token — we use EOS as padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Pad on the right (required for causal LM)

    # ── Step 2: Load base model with quantization ──────────────────────────────
    print("[2/5] Loading base model in 4-bit (QLoRA)...")
    # On MPS/CPU, skip 4-bit quantization
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,  # ← This is the QLoRA part
            device_map="auto",               # Automatically place on GPU
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # Required step before adding LoRA to a quantized model
    # This freezes the base model and prepares it for adapter training
    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    # ── Step 3: Add LoRA adapters ──────────────────────────────────────────────
    print("[3/5] Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)

    # Show how many parameters we're actually training
    model.print_trainable_parameters()
    # Expected output: "trainable params: ~5M || all params: ~3B || trainable%: ~0.17%"
    # This is the magic of LoRA — only 0.17% of parameters are updated!

    # ── Step 4: Load dataset ───────────────────────────────────────────────────
    print("\n[4/5] Loading training data...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_DATA),
            "validation": str(VAL_DATA),
        }
    )
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Val:   {len(dataset['validation'])} examples")

    # ── Step 5: Train ──────────────────────────────────────────────────────────
    print("\n[5/5] Starting training...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,              # 3 passes over the training data
        per_device_train_batch_size=2,   # 2 examples per forward pass (small for limited VRAM)
        gradient_accumulation_steps=4,  # Accumulate gradients for 4 steps before updating
                                         # Effective batch size = 2 * 4 = 8
        learning_rate=2e-4,             # Standard LoRA learning rate
        warmup_steps=10,                # Gradually increase LR for first 10 steps
        logging_steps=5,                # Log metrics every 5 steps
        evaluation_strategy="steps",    # Evaluate on val set every N steps
        eval_steps=10,
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=True,    # Keep the checkpoint with best val loss
        fp16=(device == "cuda"),        # Use fp16 only with CUDA
        report_to="none",              # Disable wandb/tensorboard for simplicity
        # For a real project: report_to="wandb" and track your experiments
    )

    # SFTTrainer = Supervised Fine-Tuning Trainer (from TRL library)
    # It handles tokenization, padding, and the training loop automatically
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",  # The field containing the formatted prompt
        max_seq_length=512,         # Max tokens per training example
    )

    print("\nTraining started. Watch the loss decrease each epoch...")
    print("(For 8 examples, this will be very fast — real datasets have 1000s of examples)\n")

    trainer.train()

    # ── Save the LoRA adapter ──────────────────────────────────────────────────
    # We ONLY save the adapter (the small trained part), NOT the full model
    # Adapter size: ~10-50MB (vs full model ~6GB)
    # To use it: load base model + load this adapter on top
    final_output = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))

    print(f"\n═══ Training Complete ═══")
    print(f"LoRA adapter saved to: {final_output}")
    print(f"Adapter size: ~{sum(p.numel() for p in model.parameters() if p.requires_grad) * 2 / 1e6:.1f}MB")
    print()
    print("To run inference with the fine-tuned model:")
    print("  python inference.py")


if __name__ == "__main__":
    main()
