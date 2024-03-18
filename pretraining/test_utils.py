import utils
import torch

"""
Test cases for functions in utils.py
"""

def test_tensor_to_list():
    tensor = torch.zeros([1, 5*5+1], dtype=torch.float32)
    tensor[0][0] = 1
    result = utils.tensor_to_list(tensor)
    print(result)

def test_list_to_tensor():
    state = [0 for i in range(5*5+1)]
    state[0] = 1
    state[1] = 1
    result = utils.list_to_tensor(state)
    print(result)

def test_check_valid():
    state = torch.zeros([1, 5*5+1], dtype=torch.float32)
    state[0][0] = 1
    state[0][1] = 1
    counter = 0
    for i in range(49):
        result = utils.check_valid(state, i)
        if result == True:
            counter+=1
            print(f"Action {i} is valid")
    
    print(counter) 

def test_get_valid_actions():
    state = torch.zeros([1, 5*5+1], dtype=torch.float32)
    state[0][0] = 1
    state[0][1] = 1
    result = utils.get_valid_actions(state)
    print(result)
    indices = [i for i, x in enumerate(result) if x == 1]
    print(indices)

def test_increment_state():
    state = torch.zeros([1, 5*5+1], dtype=torch.float32)
    state[0][0] = 1
    best_act = 49
    result = utils.increment_state(state, best_act)
    print(result)
    print(type(result))

def test_calculate_reward():
    state = torch.zeros([1, 5*5+1], dtype=torch.float32)
    state[0][0] = 1
    state[0][1] = 1
    result = utils.calculate_reward(state)
    print(result)

print("Test tensor to list")
test_tensor_to_list()

print("Test list to tensor")
test_list_to_tensor()

print("Test check valid")
test_check_valid()

print("Test get valid actions")
test_get_valid_actions()

print("Test increment state")
test_increment_state()

#print("Test calculate reward")
#test_calculate_reward()