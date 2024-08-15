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

def is_stable(configuration):
    # Get the number of rows and columns
    rows = len(configuration)
    cols = len(configuration[0])
    
    # Initialize variables to calculate the center of mass
    x_sum = 0
    y_sum = 0
    count = 0
    
    # Sum up the coordinates of all modules
    for i in range(rows):
        for j in range(cols):
            if configuration[i][j] == 1:
                x_sum += j + 0.5
                y_sum += rows - i - 0.5
                count += 1
    
    # Calculate the center of mass
    x_cm = x_sum / count
    y_cm = y_sum / count
    
    # Check stability: center of mass must lie within the base
    # The base is the bottom-most row where any module is present
    base_row = rows - 1
    base_columns = [j for j in range(cols) if configuration[base_row][j] == 1]
    
    # If no modules are in the base row, the configuration is unstable
    if not base_columns:
        return "Unstable"
    
    # The base spans from the minimum to the maximum column index in the base row
    base_min = min(base_columns)
    base_max = max(base_columns)
    
    # Check if the center of mass is within the horizontal base span
    if base_min <= x_cm <= base_max:
        return True
    else:
        return False


def generate_fixed_amount_random(n, num_robots):
    
    robots = []

    while (len(robots) < num_robots): 
        robot = np.zeros(n*n)
        num = random.randint(1, pow(2, n*n))
        
        for j in range(n*n):
            robot[j] = (num // pow(2, j)) % 2
        
        grid = utils.to_grid(robot, n=n, m=n)
            
        
        # check valid configuration
        valid = check_connected_ones(grid) #and is_stable(grid)
    
        if valid:
            robots.append(robot)
    output = open("data/configs/random10000configs.txt", "a")
    
    num_processes = int(0.85 * cpu_count())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(get_reward, robots)
    
    for result in results:
        output.write(result)

def generate_reverse():
    robots = []
    
    robot_input = open("stable45_1000.txt", "r")
    for line in robot_input.readlines():
        name, loss = (
            line.split(", ")[0],
            float(line.split(", ")[1].strip("\n"))
        )

        robots.append(np.array(name.strip('[]').split(), dtype=float))
    
    output = open("stable45_1000_reverse.txt", "a")
    
    num_processes = int(0.85 * cpu_count())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(get_reward, robots)
    
    for result in results:
        output.write(result)

def get_reward(robot):
    print (robot)
    n = int(np.sqrt(len(robot)))
    #reward, num_stairs, stair_height = utils.calculate_reward(robot, n, n)
    #return str(robot).replace("\n", "") + ", " + str(reward) + ", " + str(num_stairs) +  ", " + str(stair_height)+ "\n"

    reward = utils.calculate_reward(robot, n, 11, 1)
    return str(robot).replace("\n", "") + ", " + str(reward[0]) + "\n"

def generate_scores():
    # given file containing robot configs, create file with scores

    robots = []
    
    robot_input = open("selected_robot_configs.txt", "r")
    for line in robot_input.readlines():
        name = line.strip("\n")
        robots.append(np.array(name.strip('[]').split(), dtype=float))
    
    output = open("robot_config_sampled.txt", "a")
    
    num_processes = int(0.85 * cpu_count())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(get_reward, robots)
    
    for result in results:
        output.write(result)

#generate_reverse()   
#generate_scores()
#generate_fixed_amount_random(4, 1000)
generate_fixed_amount_random(5, 10000)
#generate_all(4)
#generate_all(5)

# use generated robots configurations to run simulations
# we want a brute force solution to check if the MCTS is working correctly

