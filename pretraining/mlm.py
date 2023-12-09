import sys
import torch
import pickle
import numpy as np
from datasets import load_metric
from tokenizer import Vocabulary, Tokenizer
from dataset import MLMDataset
from transformers import (
    BertForMaskedLM,
    BertConfig,
    BartConfig,
    BartForCausalLM,
    TrainingArguments,
    Trainer,
)

###### CONFIGS ######
TRAJECTORIES = pickle.load(open("data/trajectories.pkl", "rb"))
VOCAB = Vocabulary(9)
TOKENIZER = Tokenizer(VOCAB)
TRAIN_DATASET = MLMDataset(TRAJECTORIES[: int(len(TRAJECTORIES) * 0.8)], TOKENIZER)
TEST_DATASET = MLMDataset(TRAJECTORIES[int(len(TRAJECTORIES) - 800) :], TOKENIZER)

TRAIN_DATASET.causal_mask_tokens(0.20)
TEST_DATASET.causal_mask_tokens(0.20)

# MLM_CONFIG = BertConfig(vocab_size=len(VOCAB), type_vocab_size=1)
MLM_CONFIG = BartConfig(
    vocab_size=len(VOCAB),
    bos_token_id=VOCAB.word2idx["<sos>"],
    eos_token_id=VOCAB.word2idx["<eos>"],
    pad_token_id=VOCAB.word2idx["<pad>"],
)
MODEL = BartForCausalLM(MLM_CONFIG)
METRIC = load_metric("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return METRIC.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    training_args = TrainingArguments(
        output_dir="data/results/bert-mlm",  # output directory
        num_train_epochs=100,  # training epochs
        per_device_train_batch_size=32,  # batch size per device during training
    )

    if sys.argv[1] == "train":
        ###### TRAINING ######
        trainer = Trainer(
            model=MODEL,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=TRAIN_DATASET,  # training dataset
            eval_dataset=TEST_DATASET,  # evaluation dataset
        )

        trainer.train()
    elif sys.argv[1] == "test":
        # load the model
        MODEL = BartForCausalLM.from_pretrained("data/results/bart-causal")
        trainer = Trainer(
            model=MODEL,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=TRAIN_DATASET,  # training dataset
            eval_dataset=TEST_DATASET,  # evaluation dataset
        )
        a = trainer.predict(TEST_DATASET)

        sample_predictions = np.argmax(a.predictions, axis=2)[:10]
        sample_labels = a.label_ids[:10]

        nonzero_predictions = [v[np.nonzero(v)] for v in sample_predictions]
        nonzero_labels = [v[np.nonzero(v)] for v in sample_labels]

        detokenized_predictions = TOKENIZER.detokenize(nonzero_predictions)
        detokenized_labels = TOKENIZER.detokenize(nonzero_labels)

        for i in range(10):
            print("Prediction: ", detokenized_predictions[i])
            print("Label: ", detokenized_labels[i])
