import string
import json
from pathlib import Path
import subprocess
import sys
import torch
import numpy as np
import random
import os

sys.path.append('../')
import simulation.generate_robot_config as generate_robot_config

n_grid = 3
m_grid = 3

def tensor_to_list(tensor):
    state = [0 for _ in range(n_grid * m_grid)]
    for i in range(n_grid * m_grid):
        state[i] += tensor[i]
    return state

def list_to_tensor(state):
    #print("list to tensor", state)
    tensor = np.zeros([m_grid * n_grid], dtype=np.float32)
    for i in range(n_grid*m_grid):
        tensor[i]= int(state[i])
    assert len(state) == tensor.shape[0], f"state length is {len(state)} instead of {tensor.shape}"
    return tensor

def to_grid(tensor, n = n_grid, m=m_grid):
    grid = [[0 for i in range(n)] for j in range(m)]
    for i in range(n):
        for j in range(m):
            grid[j][i] = int(tensor[(m-j-1) * n + i])
    return grid

def to_binarystring(grid):
    state = [0 for i in range(n_grid * m_grid)]
    for i in range(n_grid):
        for j in range(m_grid):
            state[j * n_grid + i] = grid[j][i]
    return state

def check_valid(tensor, index):
    """
    tensor: tensor of length n_grid * m_grid representing robot positions in binary form
    index: index in (n_grid+2) * (m_grid+2) of the action to take
    """
    state = tensor_to_list(tensor)

    i = index % (n_grid + 2)
    j = index // (n_grid + 2)
    if index == (n_grid+2)*(n_grid +2):
        return False
        #return True
    
    if 0 < i < n_grid + 1 and 0 < j < m_grid + 1:
        if state[(i-1) + (j-1) * n_grid] == 1:
            return False
     
    
    if i == 0:
        # check column n_grid are all zeros
        for j_tmp in range(0, m_grid):
            if state[j_tmp * (n_grid) - 1] == 1:
                return False
        if j == m_grid + 1 or j == 0:
            return False
        
        if state[(j-1) * n_grid + (i-1) + 1] == 1:
            return True
        return False
            
    if i == n_grid + 1:
        # check column 0 are all zeros
        for j_tmp in range(0, m_grid):
            if state[j_tmp * n_grid] == 1:
                return False
        if j == 0 or j == m_grid + 1:
            return False
        
        if state[(j-1) * n_grid + (i-1) - 1] == 1:
            return True
        return False 

    if j == 0:
        # check row m_grid are all zeros
        for i_tmp in range(0, n_grid):
            if state[(m_grid-1) * n_grid + i_tmp] == 1:
                return False
    
        if state[(j-1) * n_grid + (i-1) + n_grid] == 1:
            return True
        return False    

    if j == m_grid + 1:
        # check row 0 are all zeros
        for i_tmp in range(0, n_grid):
            if state[i_tmp] == 1:
                return False
    
        if state[(j-1) * n_grid + (i-1) - n_grid] == 1:
            return True
        return False
    
    if j == 1 and i == 1:
        if state[(j-1) * n_grid + (i-1) + 1] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) + n_grid] == 1:
            return True
        return False
    if j == 1 and i == n_grid:
        if state[(j-1) * n_grid + (i-1) - 1] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) + n_grid] == 1:
            return True
        return False
    if j == m_grid and i == 1:
        if state[(j-1) * n_grid + (i-1) + 1] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) - n_grid] == 1:
            return True
        return False
    if j == m_grid and i == n_grid:
        if state[(j-1) * n_grid + (i-1) - 1] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) - n_grid] == 1:
            return True
        return False
    
    if j == n_grid:
        if state[(j-1) * n_grid + (i-1) + 1] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) - 1] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) - n_grid] == 1:
            return True
        return False
    if j == 1:
        if state[(j-1) * n_grid + (i-1) + 1] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) - 1] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) + n_grid] == 1:
            return True
        return False
    if i == n_grid:
        if state[(j-1) * n_grid + (i-1) - 1] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) + n_grid] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) - n_grid] == 1:
            return True
        return False
    if i == 1:
        if state[(j-1) * n_grid + (i-1) + 1] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) + n_grid] == 1:
            return True
        if state[(j-1) * n_grid + (i-1) - n_grid] == 1:
            return True
        return False


    if state[(j-1) * n_grid + (i-1) + 1] == 1:
        return True
    if state[(j-1) * n_grid + (i-1) - 1] == 1:
        return True
    if state[(j-1) * n_grid + (i-1) + n_grid] == 1:
        return True
    if state[(j-1) * n_grid + (i-1) - n_grid] == 1:
        return True
        
    return False

def get_valid_actions(tensor):
    """
    tensor: tensor of length n_grid * m_grid representing robot positions in binary form
    """
    valid_actions = np.zeros([(n_grid+2)*(n_grid +2)+1])
    for i in range((n_grid + 2)*(n_grid + 2)):
            if check_valid(tensor, i):
                valid_actions[i] = 1
    valid_actions[(n_grid+2)*(n_grid +2)] = 1
    #print (valid_actions)
    return valid_actions 

def increment_state(tensor, best_act):
    """
    tensor: binary string of length n_grid * m_grid representing robot positions
    best_act: index in (n_grid+2) * (m_grid+2) of the best action to take.
    """
    state = tensor_to_list(tensor)
    if best_act == (n_grid+2)*(n_grid +2) or best_act == -1:
        return np.append(list_to_tensor(state), 1)

    i = best_act % (n_grid + 2)
    j = best_act // (n_grid + 2)
    if i == 0:
        # shift all elements to the right
        state = [0] + state[:-1]
        state[(j-1) * n_grid] = 1

    elif i == n_grid + 1:
        # shift all elements to the left
        state = state[1:] + [0]
        state[j * n_grid - 1] = 1

    elif j == 0:
        # shift all elements up
        state = [0] * n_grid + state[:-n_grid]
        state[i - 1] = 1
        
    elif j == n_grid + 1:
        # shift all elements down
        state = state[n_grid:] + [0] * n_grid
        state[(j-1) * n_grid + i - 1] = 1
    else:
        # no shifting
        state[(j-1) * n_grid + i - 1] = 1
    assert len(state) == n_grid * m_grid, f"state length is {len(state)} instead of {n_grid * m_grid}. Length of tensor is {tensor.shape}"
    return np.append(list_to_tensor(state), 0)

def calculate_reward(state, n, obj, target_distance):
    """
    state: binary string of length n_grid * m_grid representing robot positions

    """
    #TODO: do a check here to access a dictionary to get reward

    # return 0
    
    # this block is reads lines from previously generated configurations 
    f = open("./data/configs/all_configs_rewards.txt", "r")
    print (state)
    for line in f:
        if line.split(",")[0] == str(state):
            reward = float(line.split(",")[1])
            print ("Reward form dict", reward)
            return reward, 10
    # test
    
    # transfer to json configuration
    grid = to_grid(state, n, n)
    r = generate_robot_config.Robot(grid)

    config = {
        "objects": r.objects,
        "springs": r.springs,
        "angle_springs": r.angle_springs,
    }

    binary_string = ''.join(state.astype(int).astype(str))
    name = int(binary_string, 2)
    print (name)
    json.dump(config, open(f"./robot/{name}.json", "w+"))
    
    
    # Construct the command as you would type it in the terminal
    #
    if obj == 11:
        command = ["python3", "../simulation/mass_spring.py", f"./robot/{name}.json", "train", "losses-test.json", "flat.png" ]
    elif obj == 12:
        command = ["python3", "../simulation/mass_spring.py", f"./robot/{name}.json", "train", "losses-test.json", "flat.png", "--steps=4096"]
    elif obj == 31:
        command = ["python3", "../simulation/mass_spring_reverse.py", f"./robot/{name}.json", "train", "losses-test.json", "flat.png", "--steps=4096"]
    else:
        return 0
        
    # Run the command
    command = " ".join(command)
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    # Check if the execution was successful
    if result.returncode == 0:
        print("Script executed successfully!")
    else:
        print(result.stderr)
        print("Script execution failed!") 
        return -1, -1
    
    file_path = f'./robot/{name}.txt'
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            # Read the first line
            first_line = file.readline()
    else:
        first_line = 2


    if (obj == 11 or 12):
        file_path = f'./robot/{name}x.txt'
        t = -1
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    value = float(line.strip())
                    if value > target_distance:
                        t = line_number * 0.004
                        break
                except ValueError:
                    continue  # Skip lines that cannot be converted to float
    elif (obj == 31 or obj == 32):
        file_path = f'./robot/{name}x.txt'
        t = -1
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    value = float(line.strip())
                    if value > -target_distance:
                        t = line_number * 0.004
                        break
                except ValueError:
                    continue  # Skip lines that cannot be converted to float

    if obj == 11 or obj == 12:
        return - float(first_line), t
    if obj == 31 or obj == 32:
        return 1 - float(first_line), t
    return 10000

def calculate_reward_stairs(state, n, num_stairs, target_distance, h1, h2):
    """
    state: binary string of length n * n representing robot positions

    """
    #TODO: do a check here to access a dictionary to get reward

    # return 0
    
    # this block is reads lines from previously generated configurations 
    #f = open("./data/configs/all_configs_rewards.txt", "r")
    #print (state)
    #for line in f:
    #    if line.split(",")[0] == str(state):
    #        reward = float(line.split(",")[1])
    #        print ("Reward form dict", reward)
    #        return reward
    # test
    #return 0 
    
    # transfer to json configuration
    grid = to_grid(state, n, n)
    r = generate_robot_config.Robot(grid)

    config = {
        "objects": r.objects,
        "springs": r.springs,
        "angle_springs": r.angle_springs,
    }

    binary_string = ''.join(state.astype(int).astype(str))
    name = int(binary_string, 2)
    print (name)
    json.dump(config, open(f"./robot/{name}.json", "w+"))
    
    if (num_stairs == 2):

        # Construct the command as you would type it in the terminal
        command = ["python3.8", "../simulation/mass_spring_stair.py", f"./robot/{name}.json", \
        "train", "losses-test.json", "stair.png", "-stairs 2", "-stair-widths 0.4 1.2", f"-stair-heights 0.1 {0.1-h1}"] 
    
    if (num_stairs == 3):
        # Construct the command as you would type it in the terminal
        command = ["python3.8", "../simulation/mass_spring_stair.py", f"./robot/{name}.json", \
        "train", "losses-test.json", "stair.png", "-stairs 3", "-stair-widths 0.4 0.3 1.2", f"-stair-heights 0.1 {0.1-h1} {0.1-h1-h2}"] 

    # Run the command
    command = " ".join(command)
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    # Check if the execution was successful
    if result.returncode == 0:
        print("Script executed successfully!")
    else:
        print("Script execution failed!")
        print("Error:", result.stderr)
        return -1, -1
    
    
    file_path = f'./robot/{name}.txt'

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            # Read the first line
            first_line = file.readline()
    else:
        first_line = 1

    file_path = f'./robot/{name}x.txt'
    t = -1
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                value = float(line.strip())
                if value > target_distance:
                    t = line_number * 0.004
                    break
            except ValueError:
                continue  # Skip lines that cannot be converted to float


    
    return -float(first_line), t

def calculate_reward_random_stairs(state, n):
    """
    state: binary string of length n * n representing robot positions

    """
    #TODO: do a check here to access a dictionary to get reward

    # return 0
    
    # this block is reads lines from previously generated configurations 
    #f = open("./data/configs/all_configs_rewards.txt", "r")
    #print (state)
    #for line in f:
    #    if line.split(",")[0] == str(state):
    #        reward = float(line.split(",")[1])
    #        print ("Reward form dict", reward)
    #        return reward
    # test
    #return 0 
    
    # transfer to json configuration
    grid = to_grid(state, n, n)
    r = generate_robot_config.Robot(grid)

    config = {
        "objects": r.objects,
        "springs": r.springs,
        "angle_springs": r.angle_springs,
    }

    binary_string = ''.join(state.astype(int).astype(str))
    name = int(binary_string, 2)
    print (name)
    json.dump(config, open(f"./robot/{name}.json", "w+"))
    
    num_stairs = random.randint(2, 3)
    stair_height = random.uniform(0.01, 0.05)
    if (num_stairs == 2):

        # Construct the command as you would type it in the terminal
        command = ["python3", "/home/matt/soft-lattice-robot-design/simulation/mass_spring_stair.py", f"/home/matt/soft-lattice-robot-design/pretraining/robot/{name}.json", \
        "train", "losses-test.json", "stair.png", "-stairs 2", "-stair-widths 0.4 0.8", f"-stair-heights 0.1 {0.1-stair_height}"] 
    
    if (num_stairs == 3):
        # Construct the command as you would type it in the terminal
        command = ["python3", "/home/matt/soft-lattice-robot-design/simulation/mass_spring_stair.py", f"/home/matt/soft-lattice-robot-design/pretraining/robot/{name}.json", \
        "train", "losses-test.json", "stair.png", "-stairs 3", "-stair-widths 0.4 0.3 0.4", f"-stair-heights 0.1 {0.1-stair_height} {0.1-2*stair_height}"] 

    # Run the command
    command = " ".join(command)
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    # Check if the execution was successful
    if result.returncode == 0:
        print("Script executed successfully!")
    else:
        print("Script execution failed!")
        print("Error:", result.stderr)
    
    
    file_path = f'./robot/{name}.txt'

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            # Read the first line
            first_line = file.readline()
    else:
        first_line = "5"
    return (1.5 - float(first_line)), num_stairs, stair_height


def count_occurances():
    counts = {}
    f = open("history.txt", "r")
    for line in f:
        state = (line[line.find("["):line.find("]")+1])
        state = state.replace(",", "")
        if state.count("1") == 4:
            if state in counts:
                counts[state] += 1
            else:
                counts[state] = 1
    return counts
def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation_any(x: torch.Tensor, y: torch.Tensor):
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

def spearman_correlation():
    print ("Spearman correlation")
    counts = count_occurances()
    rewards = {}
    spearman_correlation = 0
    print (counts)

    f = open("all_configs_rewards.txt", "r")
    # get reward score, and count for each leaf state
    for line in f:
        state = line.split(",")[0]
        if state in counts.keys():
            rewards[state] = float(line.split(",")[1])

    print("rewards", rewards)
    
    #get rank
    counts_rank = {}
    rewards_rank = {}

    for i, key in enumerate(sorted(counts, key=counts.get, reverse=True)):
        counts_rank[key] = i
    for i, key in enumerate(sorted(rewards, key=rewards.get, reverse=True)):
        rewards_rank[key] = i

    print (counts_rank)
    print (rewards_rank)
    
    for key in counts_rank.keys():
        print (key, counts_rank[key], rewards_rank[key])
        spearman_correlation += (counts_rank[key] - rewards_rank[key]) ** 2

    spearman_correlation = 1 - 6 * spearman_correlation / (len(counts_rank) * (len(counts_rank) ** 2 - 1))
    print(spearman_correlation)

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

