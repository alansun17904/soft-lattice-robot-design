import tqdm
import pickle
import numpy as np
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


def generate_gsl_program(path):
    """Generate a gsl program from a path, we assume that the path is valid
    and that the first block `def 0` is the first block in the path.
    """
    if len(path) == 1:
        return "def 00000"
    program = []
    vars = {node: f"{i:05d}" for i, node in enumerate(path)}
    program.extend([f"def {vars[node]}" for node in path])
    for i in range(1, len(path)):
        start_coor = path[i - 1]
        end_coor = path[i]
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


def generate_all_programs(fname, fdump):
    grids = np.load(open(fname, "rb"))
    programs = []
    for i in tqdm.tqdm(range(len(grids))):
        path = gen_shortest_path(grids[i])
        programs.append(generate_gsl_program(path))
    pickle.dump(programs, open(fdump, "wb+"))


if __name__ == "__main__":
    # get all of the environments
    N = 5
    envs = np.load("gen/data/pfinding/grids.npy")
    fig, axs = plt.subplots(N, N, figsize=(10, 10))
    for i in range(N):
        for j in range(N):
            path = gen_shortest_path(envs[i * N + j])
            # for node in path:
            #     envs[i * N + j][node] = 2
            # envs[i * N + j][path[0]] = 3
            # envs[i * N + j][path[-1]] = 3
            axs[i, j].imshow(envs[i * N + j])
            axs[i, j].axis("off")
    plt.savefig("gen/data/pfinding/no_paths_no_obstacles.pdf")
    generate_all_programs(
        "gen/data/pfinding/grids.npy", "gen/data/pfinding/programs.pkl"
    )
    # generate the gsl program for the last path
