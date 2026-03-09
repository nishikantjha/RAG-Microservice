#!/usr/bin/env python3
"""
inference.py
──────────────────────────────────────────────────────────────────────────────
Test the fine-tuned model by running inference with the LoRA adapter.

This shows the complete model loading workflow:
1. Load the base model
2. Load the LoRA adapter on top
3. Run a test prompt
──────────────────────────────────────────────────────────────────────────────
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_PATH = Path(__file__).parent / "output" / "tisix-publisher-adapter" / "final"


def generate_response(model, tokenizer, instruction: str, input_text: str = "") -> str:
    """Generate a response using the fine-tuned model."""
    if input_text.strip():
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,    # Low = deterministic
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and strip the input prompt — we only want the generated part
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    response = full_output.split("### Response:\n")[-1].strip()
    return response


def main():
    print("═══ Fine-Tuned Model Inference ═══")
    print(f"Base model : {BASE_MODEL}")
    print(f"Adapter    : {ADAPTER_PATH}")
    print()

    if not ADAPTER_PATH.exists():
        print("ERROR: Adapter not found. Run train_qlora.py first.")
        return

    # Load base model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load LoRA adapter on top of base model
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()  # Set to evaluation mode (disables dropout)

    print("Model ready!\n")

    # ── Test prompts ────────────────────────────────────────────────────────
    test_cases = [
        {
            "instruction": "Summarize the following news article in 2-3 sentences for a German publishing platform.",
            "input": "Scientists at MIT have developed a new solar panel material that is 40% more efficient than current technology. The material uses perovskite crystals and could significantly reduce the cost of solar energy production. Field tests are planned for 2026.",
        },
        {
            "instruction": "Write a compelling headline for the following article in German.",
            "input": "A Berlin-based startup has raised 20 million euros to develop AI tools specifically for small and medium-sized German businesses.",
        },
        {
            "instruction": "Extract the key facts from this article as a bullet-point list.",
            "input": "The German government announced a 200 billion euro investment plan for renewable energy infrastructure over the next decade. The plan includes 80 billion for offshore wind, 60 billion for grid modernization, and 40 billion for battery storage systems. The announcement was made at the Berlin Climate Summit attended by 45 world leaders.",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"── Test {i} ──")
        print(f"Instruction: {test['instruction']}")
        if test.get("input"):
            print(f"Input: {test['input'][:100]}...")
        print()
        response = generate_response(model, tokenizer, test["instruction"], test.get("input", ""))
        print(f"Response:\n{response}")
        print()


if __name__ == "__main__":
    main()
