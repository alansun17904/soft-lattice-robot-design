import os
import sys
import math
import json
import argparse
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from transformers import GPTNeoForCausalLM, GPT2Tokenizer 

parser = argparse.ArgumentParser()
parser.add_argument(
    "programs_file", type=str, help="file path of all of the program texts"
)
parser.add_argument("pretrain_model_name", type=str, help="pretraining model name")
parser.add_argument(
    "output_directory", type=str, help="output directory for model checkpoints"
)
parser.add_argument("-epochs", type=int, default=3)
options = parser.parse_args()

HUB_TOKEN = os.environ.get("HUB_TOKEN")
if HUB_TOKEN is None:
    print(
        "HuggingFace read/write access token is not set. \
          Please set the environmental variable HUB_TOKEN, \
          by running `export HUB_TOKEN=<hub-token>`"
    )
    sys.exit(1)

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
data = json.load(open(options.programs_file, "r"))

#tokenizer.add_tokens(["<def>", "<add>", "<n>", "<b>", "<e>", "<w>", "|"])
model.resize_token_embeddings(len(tokenizer))
# split the data into train and test
train = data[: int(0.8 * len(data))]
test = data[int(0.8 * len(data)) :]
print(f"Train: {len(train)} programs, Test: {len(test)} programs")

# tokenize all of the input sequences
train_encodings = tokenizer(train, truncation=True, padding=True, return_tensors="pt")
test_encodings = tokenizer(test, truncation=True, padding=True, return_tensors="pt")
print("Finished tokenizing")
print (type(test_encodings))
test_labels = test_encodings.copy() #tokenizer(test, truncation=True, padding=True, return_tensors="pt")
for i in range(len(test_encodings["input_ids"])):
    test_labels["input_ids"][i][:(test_labels["input_ids"][i]==50256).nonzero()[-2]] = -100
# reshape the tensors from dict to list of dict
train_encodings = [
    {
        "input_ids": train_encodings["input_ids"][i],
        "attention_mask": train_encodings["attention_mask"][i],
        "labels": torch.clone(train_encodings["input_ids"][i])
    }
    for i in range(len(train_encodings["input_ids"]))
]

test_encodings = [
    {
        "input_ids": test_encodings["input_ids"][i],
        "labels": test_labels["input_ids"][i],
        "attention_mask": torch.clone(test_encodings["attention_mask"][i])
    }
    for i in range(len(test_encodings["input_ids"]))
]



training_args = TrainingArguments(
    output_dir=options.output_directory,
    hub_token=HUB_TOKEN,
    num_train_epochs=options.epochs,
    hub_model_id=options.pretrain_model_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps = 0.2
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_encodings,
    eval_dataset=test_encodings,
)

trainer.train()
torch.save(model.state_dict(), os.path.join(options.output_directory, "model.pth"))
eval_results = trainer.evaluate()

prompt = "<|endoftext|>"
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to('cuda')
print (inputs)
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
print(tokenizer.batch_decode(outputs))
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

