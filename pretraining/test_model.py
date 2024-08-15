import torch
from transformers import pipeline
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Load the model
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#tokenizer.add_tokens(["<def>", "<add>", "<n>", "<b>", "<e>", "<w>", "|"])
model.resize_token_embeddings(len(tokenizer))


model.load_state_dict(torch.load("./all_model/aspect_ratio.pth"))
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

#prompt = "<|endoftext|> We are designing a soft modular robot for walking from left to right on a plane. Choose the better congfiguration between the following two <|endoftext|>(a)create block0; create block1; create block2; create block3; place block1 to the right of block0; place block2 to the right of block1; place block3 at the bottom of block2<|endoftext|> (b)create block0; create block1; create block2; create block3; create block4; create block5; place block1 at the bottom of block0; place block2 to the right of block0; place block3 at the bottom of block1; place block4 to the right of block3; place block5 to the left of block3<|endoftext|>"

#prompt = "<|endoftext|> We are designing a soft modular robot for walking from left to right on a plane. Choose the better congfiguration between the following two <|endoftext|>(a)create block0; create block1; create block2; create block3; place block1 at the bottom of block0; place block2 to the left of block1; place block3 at the bottom of block2<|endoftext|> (b)create block0; create block1; create block2; create block3; place block1 at the bottom of block0; place block2 at the bottom of block1; place block3 to the left of block2<|endoftext|>"

#prompt = "<|endoftext|> Please design a robot design for walking from left to right on a plane. One of the possible designs is <|endoftext|>(a)create block0; create block1; create block2; create block3; place block1 at the bottom of block0; place block2 to the left of block1; place block3 at the bottom of block2<|endoftext|>. Use at most 25 blocks, and design a better robot of this goal.<|endoftext|>"

prompt = "<|endoftext|> Please design a robot design for walking from left to right on a plane, with aspect ratio 2<|endoftext|> create block0; create block1;"
#prompt = "<|endoftext|> Please design a robot design for walking at least 0.2 distance from left to right on a plane, using at most 5 blocks<|endoftext|> create block0; create block1;"
prompt = "<|endoftext|> Please generate robot design for walking from left to right on a plane. The robot we want should have aspect ratio 0.5:<|endoftext|> create block0; create block1;"


inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
print (inputs)
outputs = model.generate(inputs, max_new_tokens=400, do_sample=True, top_k=5, top_p=0.95)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
