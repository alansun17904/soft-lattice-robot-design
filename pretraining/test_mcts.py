import torch
import numpy as np

# generate all legal robot configs mathematically to a file
def generate_all(n=3):

    output = open("all_configs.txt", "w")

    for i in range(1, pow(2, n*n)):
        # change to binary
        robot = np.zeros(n*n)
        num = i
        
        for j in range(n*n):
            robot[j] = (num // pow(2, j)) % 2
            

        
        # check valid configuration
        valid = check_valid_config(robot, n)

        if valid:
            output.write(str(robot) + "\n")

def check_valid_config(state, n=3):
    
    flag = False
    for i in range(n):
        if state[i] != 0:
            flag = True
    if flag == False:
        return False
    
    flag = False
    for i in range(n):
        if state[i *n] != 0:
            flag = True
    if flag == False:
        return False
    
    # check connectivity 
    # count number of neighbors for each cell that is not empty
    # count # of cells with 0 neighbors
    #   if only one cell -> true
    #   otherwise -> false
    # count # of cells with 1 neighbors 
    #   must be less than or equal to 2
    #   otherwise false

    counts = [-1 for _ in range(n*n)]
    
    for i in range(n*n):
        if state[i] == 0:
            continue
        else:
            count = 0
            if i % n != 0:
                count += state[i-1]
            if i % n != n-1:
                count += state[i+1]
            if i // n != 0:
                count += state[i-n]
            if i // n != n-1:
                count += state[i+n]
            counts[i] = count

    if counts.count(0) > 1:
        return False
    if counts.count(0) == 1 and  np.count_nonzero(state == 1) == 1:
        return True
    if counts.count(1) > 2:
        return False
    return True


robot = [1, 0, 0, 1, 0, 0, 1, 0, 0]

valid = check_valid_config(robot, 3)
print (valid)
generate_all(3)


# use mcts to generate