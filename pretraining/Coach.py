import torch
import os
import pickle
from MCTS import MCTS
from utils import increment_state, calculate_reward
import numpy as np
from random import shuffle
from collections import deque
import logging


log = logging.getLogger(__name__)

class Coach():
    """
    The Coach is responsible for training the MCTS.
    This is based on the implementation of ALphaGo Zero.
    https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py
    """

    def __init__(self, nnet, args):
        robot = torch.zeros([1, 3*3+1], dtype=torch.float32)
        robot[0][0] = 1
        self.nnet = nnet
        self.mcts = MCTS(robot = robot, network=nnet, args=args)
        self.trainExamplesHistory = []
        self.args = args
        self.Es = {}
    
    def executeEpisode(self):
        trainExamples = []
        episodeStep = 0
        currRobot = torch.zeros([1, 3*3+1], dtype=torch.float32)
        currRobot[0][0] = 1

        while True:
            episodeStep += 1

            # rewrite this, base on current robot get probabilities
    
            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.get_action_probabilities(currRobot, temp=temp)

            print(f'Probabilities: {pi}')
            print(f'Type of pi: {type(pi)}')

            # action = np.random.choice(len(pi), p=pi) this line is a bug so we can just run one iteration
            action = np.random.choice(len(pi), p=list(zip(*pi))[0])
            action = pi[action][1]
            currRobot = increment_state(currRobot, action)

            trainExamples.append([currRobot, pi])
            
            if currRobot[0][-1] == 1:
                if currRobot not in self.Es:
                    self.Es[currRobot] = calculate_reward(currRobot)
                r = self.Es[currRobot]
                return [(x[0], x[1], r) for x in trainExamples]


    def learn(self):
        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print(f'Starting iteration {i+1}/{self.args.numIters}')
            # examples of the iteration
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            
           
            for _ in range(self.args.numEps):
                self.mcts = MCTS(robot = torch.zeros([1, 3*3+1], dtype=torch.float32), network = self.nnet, args=self.args)  # reset search tree
                iterationTrainExamples += self.executeEpisode()
            
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            
            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.nnet.train(trainExamples)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
        
    def saveTrainExamples(self , iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, 'trainExamples_' + str(iteration) + '.pkl')
        with open(filename, "wb+") as f:
            pickle.dump(self.trainExamplesHistory, f)
        f.closed
        log.info(f'Saving trainExamples to {filename}')