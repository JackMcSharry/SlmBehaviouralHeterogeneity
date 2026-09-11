import json
import torch
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

for participant in participants:
    # Prompts each participant
    messages = [
        {
            "role": "user",
            "content": (
                f"You are a {participant}. "
                "You are participating in a behavioural experiment. "
                "You receive £10 and may give any whole number of pounds "
                "from £0 to £10 to an anonymous participant. "
                "How much do you give? Reply with only the number."
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

    # Inference does not require gradients, reducing memory use
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            # Disabled sampling for deterministic output
            do_sample=False
        )