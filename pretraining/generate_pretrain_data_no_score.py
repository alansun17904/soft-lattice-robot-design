"""
Given a list of robots expressed in their NxN binary matrix form along with
the rewards corresponding to each of these robots, we generate the GSL programs
for each of these robots for causal pretraining of the language models. 

USAGE: 
    python3 generate_pretrain_data.py <robots.json> <losses.json> <top-p> <output_file>

<robot.json> -- (str) the file containing a list of binary matrices that specify the 
robots that we wish to generate
<losses.json> -- (str) the file containing the losses from the simulation 
corresponding to each robot
<top-p> -- (float) the top percentile of robots that we want to actually generate the gsl for
<output_file> -- (str) the file that we want to output the GSL programs to

To do this, we apply the following procedure:
    1. Read in all of the binary matrices.
    2. Read in the json file `losses.json` in the simulation directory
    3. Filter the binary matrices based on the `losses.json` simulations.
Once we have all of the target binary matrices that we wish to generate GSLs for,
we then apply an exhaustive breadth-first search to get all possible ways that we can
construct this given robot. And convert this sequence of actions into the GSL text. 

The script also takes in a parameter (M) to bound the generation. If we've already
generated M data points, then it will end.
"""


import re
import json
import tqdm
import random
import argparse
import numpy as np
import math
from tabulate import tabulate


parser = argparse.ArgumentParser()
parser.add_argument(
    "input_file",
    type=str,
    help="file path of all_configs_rewards.txt file that contains the robots and rewards",
)
parser.add_argument(
    "top_p", type=float, help="the top percentile of robots that we want to keep"
)
parser.add_argument(
    "output_file",
    type=str,
    help="output file where we will export the gsl programs in json format",
)
parser.add_argument(
    "--N",
    type=int,
    help="the number of programs to generate per robot",
    default=3,
)
options = parser.parse_args()


def bfs_one_robot(robot, N=1):
    """Given an NxN binary matrix `robot` generate (non-zero entries) x N number
    of bfs search sequences over the matrix.
    """
    seqs = []
    robot = np.array(robot)
    x_start, y_start = robot.nonzero()
    for i in range(len(x_start)):
        seqs.extend(
            bfs_from_starting(robot, starting_point=(x_start[i], y_start[i]), N=N)
        )
    return seqs


def bfs_from_starting(robot, starting_point=(0, 0), N=10):
    """Given an NxN binary matrix `robot` and a starting coordinate `starting_coor`
    sample at most N unique bfs searchs from this point.
    NOTE: Assumed that `robot` is a numpy array.
    """
    seqs = set()
    count = 0
    while len(seqs) < N and count < 100 * N:
        seqs.add(sample_one_bfs(robot, starting_point))
        count += 1
    return list(seqs)


def sample_one_bfs(robot, starting_point):
    seq = []
    visited = np.zeros(robot.shape)
    q = set()
    q.add((starting_point, starting_point))
    while q:
        prev_coor, next_coor = random.sample(list(q), 1)[0]
        q.remove((prev_coor, next_coor))
        if visited[next_coor] == 1:
            continue
        seq.append((prev_coor, next_coor))
        visited[next_coor] = 1
        # add the neighbors to the queue
        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            i, j = (
                min(max(next_coor[0] + dx, 0), len(robot) - 1),
                min(max(next_coor[1] + dy, 0), len(robot) - 1),
            )
            if visited[(i, j)] == 0 and robot[(i, j)] == 1:
                q.add((next_coor, (i, j)))
    return tuple(seq)


def generate_gsl_program(seq, num_blocks_least, num_blocks_most, distance):
    """
    Given a sequence of points, generate the GSL program that will capture the
    semantics of that sequence.
    """
    block_names = {seq[0][0]: 0}
    program = [
        "create block0; ",
    ]
    for i in range(1, len(seq)):
        prev_coor, next_coor = seq[i]
        # assumed that the previous coordinate must be in the dictionary
        # otherwise this is not a valid generation, so find the name of the next_coor
        if next_coor in block_names.keys() or prev_coor not in block_names.keys():
            assert RuntimeError("Sequence of block generations is invalid.")
        block_names[next_coor] = i
        #program.extend([f"<def> block{i}"])
        program.extend([f"create block{i}; "])

        direction = get_direction(prev_coor, next_coor)
        if i == len(seq) - 1:

            program.append(
                #f"<add> block{block_names[prev_coor]} block{block_names[next_coor]} {direction}"
                f"place block{block_names[next_coor]} {direction} of block{block_names[prev_coor]}."
            )
        else:
             program.append(
                #f"<add> block{block_names[prev_coor]} block{block_names[next_coor]} {direction}"
                f"place block{block_names[next_coor]} {direction} of block{block_names[prev_coor]}; "
            )
            

    distance_round = math.floor(distance*100)/100
    
    prompt_string0 = "<|endoftext|>Please generate robot design for walking from left to right on a plane:<|endoftext|>"
    return_seq = []

    return_seq.extend (["".join([prompt_string0] + [i for i in program if i.startswith("create")] + [i for i in program if not i.startswith("create")] + ["<|endoftext|>"])])

    return return_seq


def get_direction(start, end):
    """
    Given two points, return the direction that the second point is from the
    first in terms of <n>, <s>, <e>, <w>.
    """
    if end[0] - start[0] > 0:
        #return "<b>"
        return "at the bottom"
    elif end[0] - start[0] < 0:
        #return "<n>"
        return "on the top"
    elif end[1] - start[1] > 0:
        #return "<e>"
        return "to the right"
    elif end[1] - start[1] < 0:
        #return "<w>"
        return "to the left"


def main():
    programs = []
    losses = []
    robots = []

    with open(options.input_file) as f:
        for line in f.readlines():
            # replace all `nan` values with 1000
            line = re.sub("nan", "1000", line)
            name = line.strip("\n")
            loss = 0
            losses.append((name, loss))
    #losses.sort(key=lambda x: x[0])
    for v in losses:
        
        item = (v[0].strip("[").strip("]").split(". "))
        robot = []
        for a in item:
            robot.append(int(a.replace(".", "")))
        

        row_size = int(np.sqrt(len(robot)))
        robot = [robot[x:x+row_size] for x in range(0, len(robot), row_size)]
        robot = [robot[x:x+row_size] for x in range(0, len(robot), row_size)]
        
        robots.append(robot[0])
    
    robots = [np.flip(robot, axis=0) for robot in robots]
        
    losses = [v[1] for v in losses]
    
    target_robots = sorted(zip(losses, robots), key=lambda x: x[0])[
            -int(options.top_p * len(robots)):
    ]

    print (target_robots)


    for robot in tqdm.tqdm(target_robots):
        num_blocks = np.count_nonzero(robot[1]==1)
        seqs = bfs_one_robot(robot[1], N=options.N)
        #programs.extend([generate_gsl_program(s, num_blocks, robots[0]) for s in seqs])
        max_blk_seq = random.randint(0, 4)
        iters = 0
        print (type(seqs))
        if (len(seqs) < 5):
            seqs = seqs * 5
        # Iterate over seqs
        for s in random.sample(seqs,1): 
            if (iters == 0):
                distance = robot[0]
            else:
                distance = random.uniform(0, robot[0])
            
            print (distance)
            if (max_blk_seq == iters):
                blk_input_least = num_blocks
                blk_input_most = num_blocks
            else:
                blk_input_least = random.randint(1, num_blocks)
                blk_input_most = random.randint(num_blocks, num_blocks+5)
            
            gsl_program = generate_gsl_program(s, blk_input_least, blk_input_most, distance) 
            print (gsl_program)
            programs.extend(gsl_program)
            iters += 1
    
    
    print(
        tabulate(
            [
                ["Total robots", len(robots)],
                ["Total target robots", len(target_robots)],
                ["Worst robot loss", round(target_robots[-1][0], 3)],
                ["Avg. programs per robot", len(seqs) / len(target_robots)],
                ["Total programs", len(programs)],
            ]
        )
    )

    json.dump(programs, open(options.output_file, "w+"))


if __name__ == "__main__":
    main()
