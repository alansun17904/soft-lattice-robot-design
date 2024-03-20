import string
import json
from pathlib import Path
import subprocess
import sys
import torch
import numpy as np

sys.path.append('../')
import simulation.generate_robot_config as generate_robot_config

n_grid = 3
m_grid = 3

def tensor_to_list(tensor):
    state = [0 for _ in range(n_grid * m_grid + 1)]
    for i in range(n_grid * m_grid + 1):
        state[i] += int(tensor[0][i].item())
    return state

def list_to_tensor(state):
    #print("list to tensor", state)
    tensor = torch.zeros([1, m_grid * n_grid+1], dtype=torch.float32)
    for i in range(n_grid*m_grid +1):
        tensor[0, i]= int(state[i])
    assert len(state) == tensor.shape[1], f"state length is {len(state)} instead of {tensor.shape}"
    return tensor

def to_grid(tensor):
    grid = [[0 for i in range(n_grid)] for j in range(m_grid)]
    for i in range(n_grid):
        for j in range(m_grid):
            grid[j][i] = int(tensor[0][j * n_grid + i].item())
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
    if index == (n_grid+2)*(n_grid +2)+1:
        return True
    
    if 0 < i < n_grid + 1 and 0 < j < m_grid + 1:
        if state[(i-1) + (j-1) * n_grid] == 1:
            return False
    
    
    
    if i == 0:
        # check column n_grid are all zeros
        for j_tmp in range(0, m_grid):
            if state[j_tmp * (n_grid+1) - 1] == 1:
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
    return valid_actions 

def increment_state(tensor, best_act):
    """
    tensor: binary string of length n_grid * m_grid representing robot positions
    best_act: index in (n_grid+2) * (m_grid+2) of the best action to take.
    """
    state = tensor_to_list(tensor)
    if best_act == (n_grid+2)*(n_grid +2):
        state[-1] = 1
        return list_to_tensor(state)

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
    assert len(state) == n_grid * m_grid+1, f"state length is {len(state)} instead of {n_grid * m_grid+1}. Length of tensor is {tensor.shape}"
    return list_to_tensor(state)

def calculate_reward(state):
    """
    state: binary string of length n_grid * m_grid representing robot positions

    """
    #TODO: do a check here to access a dictionary to get reward
    
    # transfer to json configuration
    grid = to_grid(state)
    r = generate_robot_config.Robot(grid)

    config = {
        "objects": r.objects,
        "springs": r.springs,
        "angle_springs": r.angle_springs,
    }

    json.dump(config, open(f"./robot/{r.name}.json", "w+"))

    # Construct the command as you would type it in the terminal
    command = ["python3.8", "../simulation/mass_spring.py", f"./robot/{r.name}.json", "train", "losses-test.json", "flat.png" ] 
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
    # Check if the execution was successful
    if result.returncode == 0:
        print("Script executed successfully!")
        #print("Output:", result.stdout)

    else:
        print("Script execution failed!")
        print("Error:", result.stderr)


    return max(-float(result.stdout.split("\n")[-2]), -20)





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