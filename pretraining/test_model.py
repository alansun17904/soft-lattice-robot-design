import torch
from transformers import pipeline
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
#from transformers import BartForCausalLM, BartTokenizer

# Load the model

#model = BartForCausalLM.from_pretrained("facebook/bart-large")
#tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
#model = BartForCausalLM.from_pretrained("facebook/bart-large")
#tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#tokenizer.add_tokens(["<def>", "<add>", "<n>", "<b>", "<e>", "<w>", "|"])
model.resize_token_embeddings(len(tokenizer))


model.load_state_dict(torch.load("./test_output/model_sentence.pth"))
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "<|endoftext|>"

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
print (inputs)
outputs = model.generate(inputs, max_new_tokens=220, do_sample=True, top_k=5, top_p=0.95)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
