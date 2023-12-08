import os
import sys
import math
import tqdm
import pickle
from torch.utils.data import TensorDataset
from transformers import BartTokenizer, BartForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


HUB_TOKEN = os.environ.get("HUB_TOKEN")
if HUB_TOKEN is None:
    print("HuggingFace read/write access token is not set.")
    sys.exit(1)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForCausalLM.from_pretrained("facebook/bart-large")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
data = pickle.load(open("data/3x3-gsl-programs.pkl", "rb"))




# split the data into train and test
train = data[: int(0.8 * len(data))]
test = data[int(0.8 * len(data)) :]
print(f"Train: {len(train)} programs, Test: {len(test)} programs")

# tokenize all of the input sequences
train_encodings = tokenizer(train, truncation=True, padding=True, return_tensors="pt")
test_encodings = tokenizer(test, truncation=True, padding=True, return_tensors="pt")
print("Finsihed tokenizing")

# reshape the tensors from dict to list of dict
train_encodings = [
    {
        "input_ids": train_encodings["input_ids"][i],
        "attention_mask": train_encodings["attention_mask"][i]
    }
    for i in range(len(train_encodings["input_ids"]))
]

test_encodings = [
    {
        "input_ids": test_encodings["input_ids"][i],
        "attention_mask": test_encodings["attention_mask"][i]
    }
    for i in range(len(test_encodings["input_ids"]))
]



training_args = TrainingArguments(
    output_dir="data/gsl-pretrained",
    hub_token=HUB_TOKEN,
    hub_model_id="gsl-pretrained-flat-ground",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_encodings,
    eval_dataset=test_encodings,
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
