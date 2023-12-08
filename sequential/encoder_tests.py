import torch
import numpy as np
from transformers import AutoConfig, BertForMaskedLM
from sequential.tokenizer import Tokenizer, Vocabulary


### LOAD MODEL ###

config = AutoConfig.from_pretrained("data/results/bert-mlm-encoder")
model = BertForMaskedLM.from_pretrained("data/results/bert-mlm-encoder")
vocabulary = Vocabulary(9)
tokenizer = Tokenizer(vocabulary)

print(vocabulary.word2idx)

### TEST MODEL ###
example_trajectories = [
    [1, 4, 7, 5, 0, 4, 0, 6, 2],  # expect lm to fill in 7
    [1, 4, 7, 5, 7, 0, 2],  # expect lm to fill in 6
]
labels = [
    [1, 4, 7, 5, 0, 4, 10, 6, 2],
    [1, 4, 7, 5, 7, 6, 2],
]
inputs = [
    {
        "input_ids": torch.LongTensor(ex).reshape(1, len(ex)),
        "labels": torch.LongTensor(lab).reshape(1, len(lab)),
    }
    for ex, lab in zip(example_trajectories, labels)
]

# run predictions
for input in inputs:
    with torch.no_grad():
        output = model(**input).logits
    # print(input["input_ids"])
    # print(np.argwhere(input["input_ids"] == 0)[-1][0])
    predicted_token = output[0, np.argwhere(input["input_ids"] == 0)].argmax(axis=-1)

    print(predicted_token)
