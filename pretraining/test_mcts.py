import torch
import numpy as np
import utils
from tqdm import tqdm
import concurrent.futures
import random
from os import cpu_count

# generate all legal robot configs mathematically to a file
def generate_all(n=3):

    output = open("all_configs_rewards.txt", "a")

    for i in tqdm(range(1, pow(2, n*n))):
        # change to binary
        robot = np.zeros(n*n)
        num = i
        
        for j in range(n*n):
            robot[j] = (num // pow(2, j)) % 2
            
        
        # check valid configuration
        valid = check_connected_ones(utils.to_grid(robot))
        #print (utils.to_grid(robot))
        #print ("valid: ", valid, robot)

        if valid:
            reward = utils.calculate_reward(robot)
            output.write(str(robot) + ", " + str(reward) + "\n")


def dfs(grid, visited, x, y):
    """ Perform DFS to mark all connected components starting from (x, y). """
    stack = [(x, y)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    while stack:
        cx, cy = stack.pop()
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid) and not visited[nx][ny] and grid[nx][ny] == 1:
                visited[nx][ny] = True
                stack.append((nx, ny))

def check_connected_ones(grid):
    """ Check if all 1's in the grid are connected using DFS. """
    flag = False
    n = len(grid)

    for i in range(n):
        if grid[n-1][i] != 0:
            flag = True
    if flag == False:
        return False
    
    flag = False
    for i in range(n):
        if grid[i][0] != 0:
            flag = True
    if flag == False:
        return False

    
    visited = np.full((n, n), False, dtype=bool)
    # Find first 1 to start DFS
    started = False
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1 and not visited[i][j]:
                if not started:
                    # Start DFS from the first 1 found
                    visited[i][j] = True
                    dfs(grid, visited, i, j)
                    started = True
                else:
                    # If there's another unvisited 1, not all 1's are connected
                    return False
    return True


def generate_fixed_amount_random(n, num_robots):
    
    robots = []

    while (len(robots) < num_robots): 
        robot = np.zeros(n*n)
        num = random.randint(1, pow(2, n*n))
        
        for j in range(n*n):
            robot[j] = (num // pow(2, j)) % 2
            
        
        # check valid configuration
        valid = check_connected_ones(utils.to_grid(robot))
        #print (utils.to_grid(robot))
        #print ("valid: ", valid, robot)
    
        if valid:
            robots.append(robot)
    
    output = open("configs_rewards_partial45.txt", "a")
    
    num_processes = int(0.85 * cpu_count())
    #num_processes = 3
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(get_reward, robots)
    
    for result in results:
        output.write(result)
        #print (result)

def get_reward(robot):
    print (robot)
    reward = utils.calculate_reward(robot)
    return str(robot).replace("\n", "") + ", " + str(reward) + "\n"
    


generate_fixed_amount_random(4, 1000)

#generate_all(4)
#generate_all(5)

# use generated robots configurations to run simulations
# we want a brute force solution to check if the MCTS is working correctly
