from Coach import Coach
from NNetWrapper import NNetWrapper
from utils import dotdict

args = dotdict({
    'numIters': 100,              # Number of training iterations.
    'numEps': 200,               # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 2,        
    'maxEpisodeLength': 5,
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'cpuct': 5,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

def main():
    nnet = NNetWrapper()
    c = Coach(nnet, args)
    c.learn()


if __name__ == "__main__":
    main()