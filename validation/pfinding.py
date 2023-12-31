import tqdm
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def gen_shortest_path(grid):
    xs, ys = np.where(grid == 0.5)
    start, end = (xs[0], ys[0]), (xs[1], ys[1])

    distances = np.zeros(grid.shape)
    distances += np.inf
    distances[start[0], start[1]] = 0

    unvisited = list(zip(*np.where(grid == 0)))
    unvisited.append(end)
    curr_node = start

    # detection of impossible task: if the unvisited list does not change
    # for 3 iterations
    while curr_node != end and len(unvisited) > 0:
        x, y = curr_node
        if curr_node != start:
            unvisited.remove(curr_node)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            i = min(max(0, x + dx), grid.shape[0] - 1)
            j = min(max(0, y + dy), grid.shape[1] - 1)
            if (i, j) in unvisited:
                distances[i, j] = min(distances[i, j], distances[x, y] + 1)
            if (i, j) == end:
                break
        curr_node = min(unvisited, key=lambda x: distances[x[0], x[1]])
    # now we have the distances from the start to the end
    # we can now backtrack to get the shortest path
    path = [end]
    curr_node = end
    while curr_node != start:
        x, y = curr_node
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            i, j = (
                min(max(0, x + dx), grid.shape[0] - 1),
                min(max(0, y + dy), grid.shape[1] - 1),
            )
            if distances[i, j] < distances[x, y]:
                path.append((i, j))
                curr_node = (i, j)
                break
    return path[::-1]


def generate_gsl_program(seq):
    """
    Given a sequence of points, generate the GSL program that will capture the
    semantics of that sequence.
    """
    block_names = {seq[0]: 0}
    program = [
        "def 0",
    ]
    for i in range(1, len(seq)):
        prev_coor, next_coor = seq[i - 1], seq[i]
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


def generate_all_programs(fname, fdump):
    grids = np.load(open(fname, "rb"))
    programs = []
    for i in tqdm.tqdm(range(len(grids))):
        path = gen_shortest_path(grids[i])
        programs.append(generate_gsl_program(path))
    json.dump(programs, open(fdump, "w+"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grids_file", type=str, help="path to the grids")
    parser.add_argument("dir", type=str, help="directory to store programs")
    parser.add_argument(
        "fname", type=str, help="filename for program texts, no extension"
    )
    options = parser.parse_args()

    envs = np.load(Path(options.dir) / Path(options.grids_file))
    # visualize 25 paths just to make sure that this is right
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            path = gen_shortest_path(envs[i * 5 + j])
            for node in path:
                envs[i * 5 + j][node] = 2
            envs[i * 5 + j][path[0]] = 3
            envs[i * 5 + j][path[-1]] = 3
            axs[i, j].imshow(envs[i * 5 + j])
            axs[i, j].axis("off")
    plt.savefig(f"{Path(options.dir) / Path(options.fname)}_preview.pdf")
    generate_all_programs(
        f"{Path(options.dir) / Path(options.grids_file)}",
        f"{Path(options.dir) / Path(options.fname)}_programs.json",
    )
