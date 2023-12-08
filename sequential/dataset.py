import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from offline_generation import total_reward


class MLMDataset(Dataset):
    def __init__(self, trajectories, tokenizer):
        self.tokenized_trajectories = tokenizer.tokenize(trajectories)
        self.padded_trajectories = None
        self.attention_masks = None
        self.masked_trajectories = None
        self.tokenizer = tokenizer
        self.pad_tokenized_trajectories()

    def pad_tokenized_trajectories(self):
        # get the max length of the trajectories
        self.max_length = max([len(t) for t in self.tokenized_trajectories])
        # pad all of the trajectories
        self.padded_trajectories = [
            t + [self.tokenizer.vocab.word2idx["<pad>"]] * (self.max_length - len(t))
            for t in self.tokenized_trajectories
        ]
        # convert to tensors
        self.padded_trajectories = torch.LongTensor(self.padded_trajectories)

        # add masks for the padding
        self.attention_masks = torch.ones_like(self.padded_trajectories)
        self.attention_masks[self.padded_trajectories == 0] = 0

    def causal_mask_tokens(self, rate):
        """
        Mask tokens from the right with `rate`.
        """
        unmasked_lens = [int(rate * len(t)) for t in self.tokenized_trajectories]
        mask = torch.LongTensor(
            [[1] * l + [0] * (self.max_length - l) for l in unmasked_lens]
        )
        self.masked_trajectories = self.padded_trajectories * mask

    def mask_tokens(self, rate):
        """
        Mask tokens in the input sequences with probability rate
        """
        rand = torch.rand(self.padded_trajectories.shape) < rate
        rand = (
            rand
            * (self.padded_trajectories != self.tokenizer.vocab.word2idx["<sos>"])
            * (self.padded_trajectories != self.tokenizer.vocab.word2idx["<eos>"])
            * (self.padded_trajectories != self.tokenizer.vocab.word2idx["<pad>"])
        )
        self.masked_trajectories = self.padded_trajectories * rand

    def __len__(self):
        return len(self.tokenized_trajectories)

    def __getitem__(self, idx):
        return {
            "input_ids": self.masked_trajectories[idx],
            "token_type_ids": torch.zeros_like(self.masked_trajectories[idx]),
            "attention_mask": self.attention_masks[idx],
            "labels": self.padded_trajectories[idx],
        }


class SequentialDataset(Dataset):
    def __init__(self, trajectories, tokenizer, sufficient_length=3):
        self.trajectories = trajectories
        self.rewards = [total_reward(trajectory) for trajectory in trajectories]
        self.tokenizer = tokenizer
        # tokenize all of the trajectories
        self.tokenized_trajectories = self.tokenizer.tokenize(self.trajectories)
        # remove trajectories that have length 0
        self.tokenized_trajectories = list(
            filter(lambda x: len(x) > sufficient_length, self.tokenized_trajectories)
        )
        # get the max length of the trajectories
        self.max_length = max([len(t) for t in self.tokenized_trajectories])
        # pad all of the trajectories
        self.padded_trajectories = [
            t + [0] * (self.max_length - len(t)) for t in self.tokenized_trajectories
        ]
        # convert to tensors
        self.padded_trajectories = torch.tensor(self.padded_trajectories)

    def __len__(self):
        return len(self.tokenized_trajectories)

    def __getitem__(self, idx):
        return self.padded_trajectories[idx], self.rewards[idx]


class AnnealingSequentialDataset(Dataset):
    """
    Annealing factor is a number in [0,1] which determines how much of
    the sequence is revealed to the model during training. For example, if

        episode : a0, a1, ... an
        annealing_factor : 0.5
        output: a0, a1, ... a(n/2)  --- (correspdoning reward)
        label: a(n/2 + 1)  --- same corresponding reward

    """

    def __init__(self, sequential_dataset, annealing_factor, maxlen=45, vocab_size=16):
        self.sequential_dataset = sequential_dataset
        self._annealing_factor = annealing_factor
        self._max_annealed_length = 45
        self._vocab_size = vocab_size
        self.prepare_annealed_seqs()

    def prepare_annealed_seqs(self):
        seqs_and_rewards = [
            self._get_annealed_sequence(i) for i in range(len(self.sequential_dataset))
        ]
        self._annealed_sequences = [seq[0] for seq in seqs_and_rewards]
        self._annealed_rewards = [seq[1] for seq in seqs_and_rewards]
        self._annealed_labels = [seq[2] for seq in seqs_and_rewards]
        self._timesteps = [seq[3] for seq in seqs_and_rewards]
        # pad the annealed sequences
        # self._max_annealed_length = max([len(s) for s in self._annealed_sequences])
        self._padded_annealed = [
            s + [0] * (self._max_annealed_length - len(s))
            for s in self._annealed_sequences
        ]
        self._attention_mask = [
            [1] * len(s) + [0] * (self._max_annealed_length - len(s))
            for s in self._annealed_sequences
        ]
        self._padded_annealed = torch.LongTensor(self._padded_annealed)

        # pad the timesteps with 0s
        self._timesteps = torch.LongTensor(self._timesteps)
        self._attention_mask = torch.tensor(self._attention_mask)
        self._timesteps = torch.LongTensor(self._timesteps)

    def _get_reward_sequence(self, seq, max_len, reward):
        # note that for every action in the sequence needs to be penalized by 1/9
        # this means that each token is penalized 1/18
        # (this maybe need to be re-thought depending on the t-type)
        return [reward - i / 18 if i <= max_len else reward for i in range(max_len)]

    def _get_annealed_sequence(self, idx):
        tokenized = self.sequential_dataset.tokenized_trajectories[idx]
        annealed = tokenized[: max(1, int(self._annealing_factor * len(tokenized)))]
        # print(tokenized)
        label = tokenized[min(len(annealed), len(tokenized) - 1)]
        corresponding_reward = torch.tensor(
            self._get_reward_sequence(
                annealed,
                self._max_annealed_length,
                self.sequential_dataset.rewards[idx],
            )
        ).reshape(self._max_annealed_length, 1)
        return (
            annealed,
            corresponding_reward,
            label,
            len(annealed),
        )

    @property
    def annealing_factor(self):
        return self._annealing_factor

    @annealing_factor.setter
    def annealing_factor(self, value):
        self._annealing_factor = value
        self.prepare_annealed_seqs()

    def __len__(self):
        return len(self.sequential_dataset)

    def __getitem__(self, idx):
        return (
            self._padded_annealed[idx],
            self._annealed_rewards[idx].float(),
            self._annealed_labels[idx],
            self._timesteps[idx],
            self._attention_mask[idx],
        )
