import torch
from transformers import pipeline, BartForCausalLM

# Load the model

model = BartForCausalLM.from_pretrained("facebook/bart-large")
model.load_state_dict(torch.load("test_output/model.pt"))
generator = pipeline("text-generation", model=model)

print (generator(prompt="Create robot structure to optimize walking from left to right\ndef 0", max_length=50))