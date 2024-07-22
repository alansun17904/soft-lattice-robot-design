import torch
import os
import pickle
from MCTS import MCTS
import utils
import numpy as np
from random import shuffle
from collections import deque
import logging
import random
from tqdm import tqdm

log = logging.getLogger(__name__)

class Coach():
    """
    The Coach is responsible for training the MCTS.
    This is based on the implementation of ALphaGo Zero.
    https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py
    """

    def __init__(self, nnet, args):
        state = np.zeros([3*3+1], dtype=np.float32)
        state[0] = 1
        self.nnet = nnet
        self.mcts = MCTS(robot = state, network=nnet, args=args)
        self.trainExamplesHistory = []
        self.args = args
        self.Es = {}
    
    def executeEpisode(self):
        trainExamples = []
        episodeStep = 0
        currState = np.zeros([3*3+1], dtype=np.float32)
        currState[0] = 1

        while True:
            episodeStep += 1

            # rewrite this, base on current robot get probabilities
    
            temp = 1 #int(episodeStep < self.args.tempThreshold)
            s = np.array2string(currState, prefix="", suffix="")
            pi = self.mcts.get_action_probabilities(currState, temp=temp)

            print(f'Probabilities: {pi}')
            print(f'Type of pi: {type(pi)}')

            #action = np.random.choice(len(pi), p=pi)  #this line is a bug so we can just run one iteration
            action = np.random.choice(len(pi), p=list(zip(*pi))[0])
            action = pi[action][1]
            trainExamples.append([currState, pi])

            currState = utils.increment_state(currState, action)
            s = np.array2string(currState, prefix="", suffix="")
            print("after action probs: ", currState)

            
            if currState[-1] == 1 or np.count_nonzero(currState) == 9:
            #if np.count_nonzero(currState) == 3:
                if s not in self.Es:
                    self.Es[s], _ = utils.calculate_reward(currState[:-1], 3, 11, 1)
                r = self.Es[s]
                return [(x[0], x[1], r) for x in trainExamples]


    def learn(self):
        #pis = np.zeros([self.args.numIters, 26])
        cor0 = np.zeros(self.args.numIters)
        cor1 = np.zeros(self.args.numIters)
        cor2 = np.zeros(self.args.numIters)
        cor3 = np.zeros(self.args.numIters)

        for i in tqdm(range(1, self.args.numIters+1)):
            # bookkeeping
            print(f'Starting iteration {i+1}/{self.args.numIters}')
            # examples of the iteration
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            
           
            for _ in range(self.args.numEps):
                self.mcts = MCTS(robot = np.zeros([3*3+1], dtype=np.float32), network = self.nnet, args=self.args)  # reset search tree
                iterationTrainExamples += self.executeEpisode()
            
            self.trainExamplesHistory.append(iterationTrainExamples)

            f = open("history.txt", "a")
            f.write("\n".join(str(element) for element in iterationTrainExamples))
            f.close()

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
            print (trainExamples[0])
            
            
            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.nnet.train(trainExamples)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            state0 = np.zeros([3*3], dtype=np.float32)
            state0[0] = 1
            state0[1] = 1
            state0[2] = 1

            cor0[i-1] = self.evaluate(state0)
            
            state1 = np.zeros([3*3], dtype=np.float32)
            state1[0] = 1
            state1[1] = 1
            state1[3] = 1

            cor1[i-1] = self.evaluate(state1)
            
            state2 = np.zeros([3*3], dtype=np.float32)
            state2[0] = 1
            state2[1] = 1
            state2[2] = 1
            state2[4] = 1

            cor2[i-1] = self.evaluate(state2)

            state3 = np.zeros([3*3], dtype=np.float32)
            state3[0] = 1
            state3[1] = 1

            cor3[i-1] = self.evaluate(state3)



        print (cor0) 
        print (cor1) 
        print (cor2) 
        print (cor3) 
        
    def saveTrainExamples(self , iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, 'trainExamples_' + str(iteration) + '.pkl')
        with open(filename, "wb+") as f:
            pickle.dump(self.trainExamplesHistory, f)
        f.closed
        log.info(f'Saving trainExamples to {filename}')

    def evaluate(self,state):
        pi, v = self.nnet.predict(state)
        valids = utils.get_valid_actions(state)

        rewards = np.zeros_like(pi)

        for i in range(25):
            if valids[i] == 1: 
                next_state = utils.increment_state(state, i) 
                rewards[i], _ = utils.calculate_reward(next_state, 3, 11, 1)

        eps = 0.0000001

        cor = (utils.spearman_correlation_any(torch.from_numpy((pi * valids)[(pi*valids)!=0]), torch.from_numpy(rewards[rewards!=0])))
        return cor

