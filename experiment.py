import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Load participants from config JSON file
with open("config/participants.json", "r", encoding="utf-8") as file:
    participants = json.load(file)

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto"
)

results = []

for participant in participants:
    # Prompts each participant
    messages = [
        {
            "role": "user",
            "content": (
                f"You are simulating this participant: {participant}. "
                "Behavioural task: You have £10. Decide how many whole pounds to give "
                "to an anonymous participant. "
                "Valid responses are: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10. "
                "Return exactly one valid integer and nothing else."
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            # Disabled sampling for deterministic output
            do_sample=False
        )

    # Decode only newly generated tokens (excludes original prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    # Store and print the participants responses (and validity)
    raw_response = response.strip()

    try:
        parsed_response = int(raw_response)
        valid = 0 <= parsed_response <= 10
    except ValueError:
        parsed_response = None
        valid = False
    
    results.append({
        "participant": participant,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "valid": valid
    })
    
    print(
        f"{participant}: {raw_response}"
        f"{'' if valid else ' [INVALID]'}"
    )

# Save responses and metadata to a timestamped directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_directory = Path("results") / timestamp
run_directory.mkdir(parents=True, exist_ok=True)

results_path = run_directory / "baseline_results.csv"
metadata_path = run_directory / "metadata.json"

with open(results_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "participant",
            "raw_response",
            "parsed_response",
            "valid"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

metadata = {
    "experiment_name": "baseline_participants",
    "timestamp": timestamp,
    "model": MODEL_NAME,
    "participant_config": "config/participants.json",
    "generation": {
        "do_sample": False,
        "max_new_tokens": 10
    },
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print(f"\nResults saved to: {run_directory}")