import pickle
import random
import torch
import torch.nn as nn
from tokenizer import Vocabulary, Tokenizer
from dataset import SequentialDataset


# TODO: actually output more than the first token or else just making a greedy choice
class BertRLModelSimpleDecoder(torch.nn.Module):
    def __init__(self, encoder, vocab_size, hidden_size, dropout):
        super().__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Bilinear(hidden_size, 1, hidden_size)

    def forward(self, x, scores_to_go, **kwargs):
        with torch.no_grad():
            x = self.encoder(x)
            x = x[0][:, 0, :]
        x = self.linear(x, scores_to_go)
        return x


class DecisionTransformer(torch.nn.Module):
    def __init__(self, decoder, vocab_size, hidden_size, max_length=40):
        super().__init__()
        self.transformer = decoder
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed_timestep = nn.Embedding(max_length, hidden_size)
        self.embed_rewards = nn.Linear(1, hidden_size)
        self.embed_actions = nn.Embedding(vocab_size, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action = nn.Sequential(
            *[nn.Linear(hidden_size, self.vocab_size), nn.Softmax(dim=-1)]
        )
        self.predict_reward = nn.Linear(hidden_size, 1)

    def forward(self, actions, scores_to_go, timesteps, attention_mask=None, **kwargs):
        bs, seq_len = actions.shape[0], actions.shape[1]
        time_embeds = self.embed_timestep(timesteps)
        reward_embeds = self.embed_rewards(scores_to_go) + time_embeds
        action_embeds = self.embed_actions(actions) + time_embeds

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones(
                (bs, seq_len), dtype=torch.long, device=action_embeds.device
            )

        # make the sequence look like [rtg1, action1, rtg2, action2, ...]
        stacked_inputs = (
            torch.stack([reward_embeds, action_embeds], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(bs, 2 * seq_len, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(bs, 2 * seq_len)
        )

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]
        x = x.reshape(bs, seq_len, 2, self.hidden_size).permute(0, 2, 1, 3)
        return self.predict_action(x[:, 1])


if __name__ == "__main__":
    # test decision transformer
    from transformers import GPT2Model, GPT2Config

    hidden_size = 144
    config = GPT2Config(
        vocab_size=16,
        n_embd=hidden_size,
    )
    gpt = GPT2Model(config)
    dt = DecisionTransformer(gpt, 16, hidden_size)
    scores_to_go = torch.randn(32, 40, 1)
    timesteps = torch.LongTensor([[random.randint(0, 39)] for _ in range(32)])
    print(timesteps.shape)
    actions = torch.randint(16, (32, 40))
    print(dt(actions, scores_to_go, timesteps).shape)
