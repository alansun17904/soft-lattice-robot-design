import sys
import tqdm
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from model import DecisionTransformer
from torch.utils.data import DataLoader
from tokenizer import Vocabulary, Tokenizer
from transformers import GPT2Config, GPT2Model
from dataset import SequentialDataset, AnnealingSequentialDataset


TRAJECTORIES = pickle.load(open("data/valid_trajectories.pkl", "rb"))
VOCAB = Vocabulary(9)
TOKENIZER = Tokenizer(VOCAB)
BASE_DATASET = SequentialDataset(TRAJECTORIES, TOKENIZER)
DATASET = AnnealingSequentialDataset(BASE_DATASET, 0.9)  # starting annealing factor
HIDDEN_SIZE = 144  # projected space dimension
GPT_CONFIG = GPT2Config(
    vocab_size=len(VOCAB),
    n_embd=HIDDEN_SIZE,
)
GPT = GPT2Model(GPT_CONFIG)
MODEL = DecisionTransformer(GPT, len(VOCAB), HIDDEN_SIZE)
DATALOADER = DataLoader(
    DATASET, batch_size=64, shuffle=False
)  # model minimizes average batch size

criterion = torch.nn.CrossEntropyLoss()


def pad_predict_actual_seqs(predicted, actual, pad_token=0):
    predicted_max_len = len(predicted[0])
    # assumed that all actual sequences are the same length
    return F.pad(actual, (0, predicted_max_len - len(actual[0])), value=pad_token)


def train_one_epoch(model, dataloader, optimizer, criterion, device, losses):
    model.train()
    bar = tqdm.tqdm(dataloader, desc="Train")
    for batch in dataloader:
        optimizer.zero_grad()
        annealed_seq = batch[0].to(device)  # annealed sequence
        rtg_seq = batch[1].to(device)  # rtg sequence
        timesteps = batch[3].to(device)  # timestep sequence
        labels = batch[2].to(device)
        attention_mask = batch[4].to(device)

        timesteps = timesteps.reshape(len(timesteps), 1)
        output = model(annealed_seq, rtg_seq, timesteps)

        # only get the first timestep of the output
        loss = torch.sum(criterion(output[:, 0, :], labels))
        loss.backward()
        optimizer.step()
        bar.set_postfix(loss=loss.item())
        losses.append(loss.cpu().detach().item())
        bar.update()


def test(model, device, max_steps=30, num_paths=64, treward=0, topk=3):
    model.eval()
    paths = torch.zeros((num_paths, max_steps), dtype=torch.long).to(device)
    paths[:, 0] = 0  # always start with the <add> token
    # robot positions masking
    def mask_rp(paths, i):
        """Mask robot positions, so everything after token 6"""
        # note that we should make this dynamic to the vocab size instead of
        # setting this manually
        rp_mask = torch.zeros_like(paths)
        rp_mask[:, i, 3:] = 1
        return (1 - rp_mask) * paths

    # action functions masking
    def mask_af(paths, i):
        """Mask action functions, so everything before token 6"""
        af = torch.zeros_like(paths)
        af[:, i, :3] = 1
        return (1 - af) * paths

    scores_to_go = treward * torch.ones((num_paths, max_steps, 1)).to(device)
    attention_mask = torch.zeros((num_paths, max_steps)).to(device)
    timestep = torch.zeros((num_paths, 1), dtype=torch.long).to(device)
    attention_mask[:, 0] = 1
    # note: model input is annealed sequence, rtg sequence, timestep sequence
    # annealed_seq: [bs, max_steps, vocab_size]
    # rtg_seq: [bs, max_steps, 1]
    # timestep_seq: [bs, max_steps]
    with torch.no_grad():
        for i in tqdm.tqdm(range(1, max_steps)):
            attention_mask[:, i] = 1
            # on even steps, we mask the robot positions, on odd steps we mask the
            # action functions.
            timestep[:, 0] = i
            output = model(paths, scores_to_go.float(), timestep, attention_mask).to(
                device
            )
            if i % 2 == 0:
                output = mask_rp(output, 0)
            else:
                output = mask_af(output, 0)
            # output = torch.argmax(output[:,-1,:], dim=1)
            # output = torch.argmax(output, dim=-1)
            # make everything 0 except for the topk in output
            topk_output = torch.topk(output[:, 0], topk, dim=1)
            topk_mask = torch.zeros_like(output)
            topk_mask[topk_output.indices] = 1
            output = torch.multinomial(output[:, 0], 1)
            paths[:, i] = output.flatten()
    return paths


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL.to(device)
    print(len(VOCAB))
    if len(sys.argv) == 1 or sys.argv[1] == "train":
        losses = []
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        for epoch in range(50):
            DATASET.annealing_factor = 0.9 * (
                1 - epoch / 50
            )  # worth experimenting with: trying different annealing functions
            train_one_epoch(model, DATALOADER, optimizer, criterion, device, losses)
            # paths = test(model, device)
            # pickle.dump(paths, open(f"data/paths-epoch-{epoch}.pkl", "wb+"))
            pickle.dump(losses, open(f"data/losses-epoch-{epoch}.pkl", "wb+"))
            if epoch % 5 == 0:
                torch.save(model.state_dict(), f"data/chkpt/dt-{epoch}.pth")
                print("saved torch model")
    elif sys.argv[1] == "test":
        if len(sys.argv) < 3:
            print("Usage: python train.py test <model_path> <output_path>")
            sys.exit(1)
        path = sys.argv[2]
        model.load_state_dict(torch.load(path, map_location=device))
        paths = test(model, device)
        if len(sys.argv) == 4:
            output_path = sys.argv[3]
        else:
            output_path = "data/test-paths.pkl"
        pickle.dump(paths, open(output_path, "wb+"))
