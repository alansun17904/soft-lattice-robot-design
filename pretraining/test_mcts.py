# test mcts nnet temp/best.pth.tar

from NNetWrapper import NNetWrapper
from Coach import Coach
from pretrain_mcts import args
import numpy as np
import utils
import torch
import matplotlib.pyplot as plt

def main():
    nnet = NNetWrapper()
    nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')

    print (type(nnet)) 
    state = np.zeros([3*3], dtype=np.float32)
    state[0] = 1
    state[1] = 1
    state[3] = 1
    state[6] = 1


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
    im = ax.imshow(pi[0:25].reshape((5, 5)))
    plt.savefig("heatmap.png")
    

    


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



main()
