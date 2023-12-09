"""
Randomly generate structures in a 3x3 grid using GSL syntax for offline training.
"""


import json
import random
import pickle
from tabulate import tabulate


N = 100000
ROBOT_SIDE = 5
HASHED = json.load(open("data/3x3-t-invariant-mapping.json", "r"))


def gen_one_robot():
    # choose a random coordinate
    visited = [[0 for _ in range(ROBOT_SIDE)] for _ in range(ROBOT_SIDE)]
    num_blocks = random.randint(1, ROBOT_SIDE * ROBOT_SIDE)
    coor = (random.randint(0, ROBOT_SIDE - 1), random.randint(0, ROBOT_SIDE - 1))
    seq = []
    queue = [(coor, coor)]
    while num_blocks > 0:
        start_coor, end_coor = queue.pop(random.randint(0, len(queue) - 1))
        if visited[end_coor[0]][end_coor[1]] == 1:
            continue
        seq.append((start_coor, end_coor))
        visited[end_coor[0]][end_coor[1]] = 1
        # add the neighbors to the queue
        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            i, j = (
                min(max(end_coor[0] + dx, 0), ROBOT_SIDE - 1),
                min(max(end_coor[1] + dy, 0), ROBOT_SIDE - 1),
            )
            if visited[i][j] == 0:
                queue.append((end_coor, (i, j)))
        num_blocks -= 1
    return seq


def generate_gsl_program(seq):
    """
    Given a sequence of points, generate the GSL program that will capture the
    semantics of that sequence.
    """
    program = []
    vars = {coor[1]: f"{i:05d}" for i, coor in enumerate(seq)}
    program.extend([f"def {k}" for k in vars.values()])
    for act in seq[1:]:
        start_coor = act[0]
        end_coor = act[1]
        direction = get_direction(start_coor, end_coor)
        program.append(f"add {vars[start_coor]} {vars[end_coor]} {direction}")
    return "\n".join(program)


def get_direction(start, end):
    """
    Given two points, return the direction that the second point is from the
    first in terms of <n>, <s>, <e>, <w>.
    """
    if end[0] - start[0] > 0:
        return "<s>"
    elif end[0] - start[0] < 0:
        return "<n>"
    elif end[1] - start[1] > 0:
        return "<e>"
    elif end[1] - start[1] < 0:
        return "<w>"


def main():
    """Generate a list of GSL programs and map them to their corresponding
    robot configuration index.
    """
    programs = []
    for i in range(N):
        seq = gen_one_robot()
        programs.append(generate_gsl_program(seq))
    # dump all of the programs into a file
    print(f"Generated {len(programs)} programs.")
    pickle.dump(programs, open("data/3x3-gsl-programs.pkl", "wb"))


if __name__ == "__main__":
    main()
