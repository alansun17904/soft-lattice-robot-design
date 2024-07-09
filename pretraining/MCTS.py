import gc
import torch
import torch.nn.functional as F
import random
import utils
import math
import numpy as np
import multiprocessing
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

"""
How to represtent the state of the robot:
bit string of length n by n where n is width and m is height of the grid
1 means there is a block at that position
0 means there is no block at that position

n(m-1)  .   .   .   .   n^2 - 1
.   .   .   .   .   ... .
.   .   .   .   .   ... .
n*2 n*2+1   .   .   ... .
n   n+1 n+2 n+3 n+4 ... 2n-1
0   1   2   3   4   ... n-1

action would be position of the block to add
and the mode of the action 

the robot is automatically pinned to the bottom left corner
determine possible positions to place new blocks
Two issues:
1. same position but different edges
    For example 2x2 grid, current state is 1110.
    Possible positions of adding a block are edges 1 2 3 4 5 6 7 8,
    but 4 and 5 are the same position
2. how to determine same state -> the robot would be pinned to bottom left corner
"""

n_grid = 3
EPS = 1e-8

class MCTS:
    """
    Using Monte-Carlo tree search find the best action to take at any given
    time step: add block at certain position or stopping. This is based on the
    implementation of AlphaGo Zero.
    https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py

    Definitions:
        s: state. robot state
        a: action (index, mode) 
                    index: index of the block that is being added (shift doesn't need to be defined, the index will give the shift)
                    mode: 0 for adding a block, 1 for stopping

        Ps: policy vector. Probabilities for each action. (Z, mode) where Z is
            the "interaction" probability.
        Ns: number of times state s was visited.
        Nsa: number of times state s was visited while taking action a.
        Qsa: action-value function for state s and action a.
        Usa: upper confidence bound for action a in state s.
    """

    def __init__(self, robot, network, args, num_simulations=50, max_depth=None, eps=1e-8,gpu=False):
        """
        :param robot: the starting robot to begin the search from
        :param network: the neural network used to evalate
        """
        self.eps = eps
        self.robot = robot
        self.num_modules = np.count_nonzero(robot)
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.device = torch.device("cuda" if gpu else "cpu")
        self.network = network
        self.args = args
        
        if self.max_depth is None:
            self.max_depth = n_grid * n_grid

        self.Qsa = {}
        self.Nsa = {}   # state action visit count
        self.Ns = {}    # state visit count
        self.Ps = {}    # stores initial policy (returned by neural net)
        self.Vs = {}    # valids actions of the state
        self.Es = {}    # end states

    def get_action_probabilities(self, state, temp=1):
        """
        This function performs simulations of MCTS starting from
        state `state`. Then, the action probabilities from the root state
        are calculated and returned. The temperature parameter determines
        the level of exploration vs exploitation. If temp is 0, then the
        action with the highest visit count is chosen (fully determinisitc).
        If temp is 1, then the probabilities are distributed according to visit count.

        state: numpy array representing the robot state

        """
        # print ("get_action_probabilities") 
        #m = multiprocessing.Manager()
        #lock = m.Lock()
        #with ProcessPoolExecutor(max_workers=4) as executor:
            #print ("executor")
            #  from the current state until a leaf node is found or
            # the maximum depth is reached which ever comes first.
        #    futures = [None for _ in range(self.num_simulations)]
        
        for i in range(self.num_simulations):
            #    futures[i] = executor.submit(self.search, state, lock=lock)
            search_val = self.search(state)


        next_state_actions_arr = utils.get_valid_actions(state)
        next_state_actions = [i for i, x in enumerate(next_state_actions_arr) if x == 1]
        
        s = np.array2string(state, prefix="", suffix="")

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in next_state_actions]

        searched_action_space = [a for a in next_state_actions]
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return list(zip(probs, searched_action_space))

        counts = [x ** (1.0 / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]

        return list(zip(probs, searched_action_space))

    def Usa(self, state, mode_probs, action, c=1):
        index, mode = action
        mode_probs = mode_probs.flatten()
        if index is None:
            Ps = mode_probs[1]
        else:
            Ps = mode_probs[mode]
            # check if the action is valid
            if mode == 0 and not utils.check_valid(state, index):
                Ps = 0
        return (
            c
            * Ps
            * math.sqrt(self.Ns.get(state, self.eps))
            / (1 + self.Nsa.get((state, action), 0))
        )

    def mask_invalid_actions(self, state, add):
        """
        :param: add is a tensor of length N_grid by N_grid representing the action probabilities
        """
        for i in range(len(add)):
            if utils.check_valid(state, i):
                add[i] = 0
        return add

    def search(self, state, stop=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound: Q(s,a) + cP(s,a) / (1 + N(s,a)).

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        """
        # print ("search function") 
        robot_count = np.count_nonzero(state)
        t = torch.tensor([robot_count])  # .to(self.device)
        s = np.array2string(state, prefix="", suffix="")
        # check robot is in end state

        #if s[0][-1] == 1:
        if robot_count == 4:
            #print("end state")
            if s not in self.Es:
                self.Es[s], _ = utils.calculate_reward(state, 3, 11, 1)
            return self.Es[s]
    

        if s not in self.Ps:
            # print ("s not in self.Ps")
            self.Ps[s], v = self.network.predict(state)
            valids = utils.get_valid_actions(state)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 1
            return v


        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(5 * 5+1):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        # print("best_act", a)
        next_s = utils.increment_state(state, a)
        
        # print("next_s", next_s)
        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = max(v, self.Qsa[(s,a)])#(self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)# 
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v

        ## This is leaf node
        #if depth >=self.max_depth or state[-1][1] == 1 and robot_count != 0:
        #    return self.network(g, t)[2].flatten()[0]
        ## generate all the valid actions from the current state
        ## as well as the action probabilites from the neural network.
        #action, value = self.network(g) 
        ## target = target.cpu()
        ## action = action.cpu()
        ## value = value.cpu()

        #with torch.no_grad():
        #    #if lock is not None:
        #    #    print ("lock")
        #    #    lock.acquire()
        #    if state not in self.Ns:
        #        print("hereeeee")
        #        self.Ns[tuple(state)] = 0
        #        return value.flatten()[0]
        #    # find u for doing nothing
        #    u_pass = self.Qsa.get((*state, (None, 2)), 0) + self.Usa(
        #        state, target, action, (None, 2)
        #    )
        #    if lock is not None:
        #        lock.release()
        #    cur_best = u_pass
        #    best_act = (None, 2)
        #    # find u for all the actions that we've explored
        #    if lock is not None:
        #        lock.acquire()
        #    visited = list(
        #        filter(
        #            lambda x: (x[0][-1][2] == robot_count) and x[1][0] is not None,
        #            self.Qsa.keys(),
        #        )
        #    )
        #    u_visited = torch.Tensor(
        #        [
        #            self.Qsa.get(v, 0) + self.Usa(state, target, action, v[1])
        #            for v in visited
        #        ]
        #    )
        #    if lock is not None:
        #        lock.release()
        #    if len(u_visited) > 0:
        #        visited_max_val = torch.max(u_visited)
        #        visited_max_arg = torch.argmax(u_visited)

        #        if visited_max_val > cur_best:
        #            cur_best = visited_max_val
        #            best_act = visited[visited_max_arg][1]
        #    # assign 0 to all actions taken, sort Ps and then take greatest
        #    add = torch.clone(target)

        #    add = self.mask_invalid_actions(self.robot, state, add)

        #    if lock is not None:
        #        lock.acquire()
        #    add_max_val = (
        #        action.flatten()[0] * add.max() * math.sqrt(self.Ns[state] + self.eps)
        #    )
        #    if lock is not None:
        #        lock.relase()

        #    add_max_arg = torch.argmax(add.flatten())

        #    if add_max_val > cur_best:
        #        cur_best = add_max_val
        #        best_act = (add_max_arg, 0)


        #    # increment the current state
        #    best_act = (
        #        int(best_act[0]) if best_act[0] is not None else None,
        #        best_act[1],
        #    )
        #    next_state = utils.increment_state(state, best_act)
        #    v = self.search(next_state, depth + 1)

        #    if lock is not None:
        #        lock.acquire()
        #    if (state, best_act) in self.Qsa:
        #        self.Qsa[(state, best_act)] = (
        #            self.Nsa[(state, best_act)] * self.Qsa[(state, best_act)] + v
        #        ) / (self.Nsa[(state, best_act)] + 1)
        #        self.Nsa[(state, best_act)] += 1
        #    else:
        #        self.Qsa[(state, best_act)] = v
        #        self.Nsa[(state, best_act)] = 1
        #    self.Ns[state] += 1
        #    if lock is not None:
        #        lock.release()
        #    return v