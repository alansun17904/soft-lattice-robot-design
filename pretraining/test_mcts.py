# test mcts nnet temp/best.pth.tar

from NNetWrapper import NNetWrapper
from Coach import Coach
from pretrain_mcts import args
import numpy as np
import utils
import torch
import matplotlib.pyplot as plt
import re

def main():
    nnet = NNetWrapper()
    nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')

    print (type(nnet)) 
    state = np.zeros([3*3], dtype=np.float32)
    state[0] = 1
    state[1] = 1


    pi, v = nnet.predict(state)
    print (pi, v)
    
    valids = utils.get_valid_actions(state)
    
    rewards = np.zeros_like(pi)

    for i in range(25):
        if valids[i] == 1:
            next_state = utils.increment_state(state, i)
            rewards[i], _ = utils.calculate_reward(next_state, 3, 11, 1)
    print (pi)
    print (rewards)
    print (valids)
    eps = 0.0000001

    print ((pi * valids)[rewards != 0])
    print (rewards[rewards != 0])
    
    print (spearman_correlation(torch.from_numpy((pi * valids)[(pi*valids)!=0]), torch.from_numpy(rewards[rewards!=0])))

    fig, ax = plt.subplots()
    im = ax.imshow((pi * valids)[0:25].reshape((5, 5)))
    ax.invert_yaxis()

    cbar = ax.figure.colorbar(im, ax=ax)

    plt.title("Policy")
    plt.savefig("heatmap_pi.png")
    
    fig, ax = plt.subplots()
    im = ax.imshow(rewards[0:25].reshape((5, 5)))
    ax.invert_yaxis()
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.title("Rewards")
    plt.savefig("heatmap_rewards.png")
    plt.show()

    file_path = 'model_eval.txt'
    sequences = read_sequences_from_file(file_path)
    #plot_sequences(sequences)


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)

def read_sequences_from_file(file_path):
    sequences = []
    current_sequence = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Remove leading and trailing whitespace
            line = line.strip()
            if line:
                # Add numbers to the current sequence
                current_sequence.extend(re.findall(r"[-+]?\d*\.\d+|\d+", line))
                
                # If the line ends with ']', it marks the end of a sequence
                if line.endswith(']'):
                    # Convert the current sequence to a numpy array and add to sequences
                    sequences.append(np.array(current_sequence, dtype=float))
                    # Clear current sequence for the next one
                    current_sequence = []
    return sequences

def plot_sequences(sequences, y_label='Spearman correlation', x_label='Iteration'):
    plt.figure(figsize=(10, 5))
    for i, sequence in enumerate(sequences):
        plt.plot(sequence, marker='o', label=f'Sequence {i+1}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label}')
    plt.legend()
    plt.grid(True)
    plt.show()


main()
