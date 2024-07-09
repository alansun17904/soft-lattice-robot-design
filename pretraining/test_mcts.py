# test mcts nnet temp/best.pth.tar

from NNetWrapper import NNetWrapper
from Coach import Coach
from pretrain_mcts import args
import numpy as np

def main():
    nnet = NNetWrapper()
    nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')

    print (type(nnet)) 
    state = np.zeros([3*3], dtype=np.float32)
    state[0] = 1
    state[1] = 1
    state[3] = 1
    pi, v = nnet.predict(state)
    print (pi, v)

main()