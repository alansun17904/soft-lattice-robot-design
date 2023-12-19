import random
from typing import Tuple, List
import numpy as np


def gen_space_filling(environment: np.array, start: Tuple[int], n: int,reward=None) -> List[str]:
    programs = []
    for _ in range(n):
        seq = get_visit_sequence(environment, start, reward)
        programs.append(generate_gsl_program(seq))
    return programs


def get_visit_sequence(environment: np.array, start_coor, rewards=None):
    """
    Given the environment, get a random sequence of points that
    visits all of the interior points of the environment.
    """
    cgrid = np.copy(environment)
    seq = []
    queue = [(start_coor, start_coor)]
    while len(queue) > 0:
        # shuffle the queue and then pop the first item
        start, end = queue.pop(random.randint(0, len(queue) - 1))
        if cgrid[end[0], end[1]] == 1:
            continue
        cgrid[end[0], end[1]] = 1
        seq.append((start, end))
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            i, j = (
                min(max(0, end[0] + dx), environment.shape[0] - 1),
                min(max(0, end[1] + dy), environment.shape[1] - 1),
            )
            if cgrid[i, j] == 0:
                if rewards is None:
                    queue.append(((end[0], end[1]), (i, j)))
                if rewards is not None and rewards[i,j] >= 0:
                    queue.append(((end[0], end[1]), (i, j)))
    return seq


def generate_gsl_program(seq):
    """
    Given a sequence of points, generate the GSL program that will capture the
    semantics of that sequence.
    """
    block_names = {seq[0][0]: 0}
    program = [
        "def 0",
    ]
    for i in range(1, len(seq)):
        prev_coor, next_coor = seq[i]
        # assumed that the previous coordinate must be in the dictionary
        # otherwise this is not a valid generation, so find the name of the next_coor
        if next_coor in block_names.keys() or prev_coor not in block_names.keys():
            assert RuntimeError("Sequence of block generations is invalid.")
        block_names[next_coor] = i
        program.extend([f"def {i}"])

        direction = get_direction(prev_coor, next_coor)
        program.append(
            f"add {block_names[prev_coor]} {block_names[next_coor]} {direction}"
        )
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


if __name__ == "__main__":
    import tqdm
    import json
    import argparse
    from pathlib import Path
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("grids_file", type=str, help="path to the grids")
    parser.add_argument("starts_file", type=str, help="path to starting points")
    parser.add_argument("dir", type=str, help="directory to store programs")
    parser.add_argument(
        "fname", type=str, help="filename for program texts, no extension"
    )
    parser.add_argument("-rewards_path", type=str, help="path to rewards matrix")
    options = parser.parse_args()

    # get all of the saved environments
    grids = np.load(Path(options.dir) / Path(options.grids_file))
    starts = np.load(Path(options.dir) / Path(options.starts_file))
    if options.rewards_path is not None:
        rewards = np.load(Path(options.dir) / Path(options.rewards_path))
    else:
        rewards = None

    # convert each of the starts to tuples
    starts = [tuple(s.flatten().tolist()) for s in starts]

    # generate 1 program for each environment
    programs = []
    plength = []
    for i, (grid, start) in tqdm.tqdm(enumerate(zip(grids, starts))):
        if rewards is not None:
            program = gen_space_filling(grid, start, 1, rewards[i])
        else:
            program = gen_space_filling(grid, start, 1)
        plength.append(len(program[0].split("\n")))
        programs.append(program)

    # save the programs
    json.dump(programs, open(f"{Path(options.dir) / Path(options.fname)}_programs.json", "w+"))
    plt.hist(plength, bins=100)
    plt.savefig(f"{Path(options.dir) / Path(options.fname)}_program_lengths.pdf")
