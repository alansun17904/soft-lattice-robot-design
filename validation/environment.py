import tqdm
import json
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def rand_env_empty_generation(n: int, output_end_coor=True) -> np.array:
    """Creates an environment which is an nxn binary matrix. Then, randomly
    pick two coordinates in the matrix which are respectively called the
    start and the end.
    """
    start = (random.randint(0, n - 1), random.randint(0, n - 1))
    end = (random.randint(0, n - 1), random.randint(0, n - 1))
    while end == start:
        end = (random.randint(0, n - 1), random.randint(0, n - 1))
    grid = np.zeros((n, n))
    grid[start] = 0.5
    if output_end_coor:
        grid[end] = 0.5
        return grid, start, end
    else:
        return grid, start


def rand_env_generation_diff_rewards(n: int, pts: int = 1) -> np.array:
    """Creates an environment which is an nxn real matrix (with a closed contour).
    Then, for every point inside of the contour, is assigned a number between [-1,1]
    whose value represents the reward if the model places a block there.
    """
    contour, start, interior = rand_env_generation(n, pts)
    rewards = [2 * random.random() - 1 for _ in range(len(interior))]
    reward_grid = np.zeros((n, n))
    for (x, y), r in zip(interior, rewards):
        reward_grid[x, y] = r
    return contour, start, interior, reward_grid


def rand_env_generation(n: int, pts: int = 1) -> np.array:
    """Generates an environment which is an nxn binary matrix. Then, create a
    closed contour in the matrix. Outputs
        - the binary matrix where the 1 coordinates represent the outline of the contour
        - a random point inside the contour
        - the set of all coordinates that are inside the contour
    """
    contour = gen_random_contour(n)
    interior = get_inside_of_contour(contour)
    # get `pts` number of random points on the interior of the contour
    return contour, random.choices(interior, k=pts), interior


def gen_random_pfilling_env(n: int, num: int, obstacle_size: int = 1) -> np.array:
    """Generate a nxn matrix with `num` obstables.
    Each obstable is an island (a continous region of 1s) with maximum
    block size of `obstable_size`.
    """
    grid = np.zeros((n, n))
    for _ in range(num):
        # generate a random island
        gen_random_island(grid, obstacle_size)
    # two random points now that are not on islands (starting and ending points)
    xs, ys = np.where(grid == 0)
    coors = list(zip(xs, ys))
    start = random.choice(list(zip(xs, ys)))
    coors.remove(start)
    end = random.choice(coors)
    grid[start] = 0.5
    grid[end] = 0.5
    return grid, start, end


def gen_random_island(grid, obstacle_size: int = 1):
    """Generate a random island on the grid with a maximum block size of
    `obstacle_size`. Note that this function modifies the `grid` in place.
    """
    island_size = random.randint(1, obstacle_size)
    # get a random starting point
    xs, ys = np.where(grid == 0)
    # choose a random starting point
    x, y = random.choice(list(zip(xs, ys)))
    # generate a random island starting from this point (do bfs starting from this point)
    queue = [(x, y)]
    while len(queue) > 0 and island_size > 0:
        x, y = queue.pop(0)
        grid[x, y] = 1
        island_size -= 1
        # add all of the neighbors to the queue
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            i, j = (
                min(max(0, x + dx), grid.shape[0] - 1),
                min(max(0, y + dy), grid.shape[1] - 1),
            )
            if grid[i, j] == 0:
                queue.append((x + dx, y + dy))
    return grid


def gen_random_contour(n: int):
    """
    Given a side length `n`, generate a binary matrix with a random contour
    where the contour one pixel wide with 1 representing the contour and 0
    representing the background.
    """
    # HYPERPARAMETERS FOR CONTOUR GENERATION
    points = 8  # the number of points in the contour
    r = 0.4  # perturbation radius
    sr = n / 3  # starting radius
    xt, yt = n / 2, n / 2  # starting coordinates
    # generate random points on the circle
    angles = np.linspace(0, 2 * np.pi, points)
    coors = np.array(
        [sr * np.cos(angles) + xt, sr * np.sin(angles) + yt]
    ).T * np.random.uniform(1 - r, 1 + r, (points, 2))
    coors = coors.astype(int)
    grid = np.zeros((n, n))

    def relocate(coor):
        coor[0] = min(max(0, coor[0]), grid.shape[0] - 1)
        coor[1] = min(max(0, coor[1]), grid.shape[1] - 1)
        return coor

    relocate(coors[0])
    grid[coors[0, 0], coors[0, 1]] = 1
    # generate the contour
    for i in range(1, len(coors)):
        start_coor = coors[i - 1]
        end_coor = coors[i] if i != len(coors) - 1 else coors[0]
        relocate(start_coor)
        relocate(end_coor)
        grid[start_coor[0], start_coor[1]] = 1
        draw_line(grid, start_coor, end_coor)
    return grid


def draw_line(grid, start, end):
    # anchor the starting and ending coordinates in the grid
    x, y = start
    while x != end[0] or y != end[1]:
        grid[x, y] = 1
        x += np.sign(end[0] - x)
        y += np.sign(end[1] - y)
    grid[end[0], end[1]] = 1
    return grid


def get_inside_of_contour(grid, start_point=None):
    """
    Given a grid with a contour denoted by 1s and the background denoted by 0s,
    find the inside of the contour and return their coordinates.

    The midpoint will always be in the interior, so BFS is performed from
    the midpoint
    """
    cgrid = np.copy(grid)
    if start_point is None:
        start_point = grid.shape[0] // 2, grid.shape[1] // 2
    queue = [start_point]
    visited = []
    while len(queue) > 0:
        x, y = queue.pop(0)
        if cgrid[x, y] == 1:
            continue
        visited.append((x, y))
        cgrid[x, y] = 1
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            i, j = (
                min(max(0, x + dx), grid.shape[0] - 1),
                min(max(0, y + dy), grid.shape[1] - 1),
            )
            if cgrid[i, j] != 1:
                queue.append((i, j))
    return visited


def create_one_random_shape(grid, start, size=None):
    if size is None:
        size = int(0.3 * len(grid) ** 2)
    # do random bfs from the startng point
    visited = []
    q = [start]
    count = 1
    while count < size and q:
        curr = q.pop(random.randint(0, len(q) - 1))
        if grid[curr] != 0.5:
            grid[curr] = 1
        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            i, j = (
                min(max(curr[0] + dx, 0), len(grid) - 1),
                min(max(curr[1] + dy, 0), len(grid) - 1),
            )
            if (i, j) not in visited and grid[(i, j)] != 1:
                q.append((i, j))
        visited.append(curr)
        count += 1
    return grid


def gen_diff_sfilling_envs(fname, grid_size=10, N=1000, _dir="gen/data/sfilling/diff"):
    agg_grids = []
    agg_pts = []
    rewards = []
    for _ in tqdm.tqdm(range(N)):
        try:
            grid, pts, _, rewrads = rand_env_generation_diff_rewards(grid_size)
        except IndexError:
            continue
        agg_grids.append(grid)
        agg_pts.append(pts)
        rewards.append(rewrads)
    np.save(f"{Path(_dir) / Path(fname)}_grids.npy", np.array(agg_grids))
    np.save(f"{Path(_dir) / Path(fname)}_pts.npy", np.array(agg_pts))
    np.save(f"{Path(_dir) / Path(fname)}_rewards.npy", np.array(rewards))

    # sample 64 random grids from all of the grids and plot them
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            target_grid = agg_grids[i * 8 + j]
            target_rewards = rewards[i * 8 + j]
            axs[i, j].imshow(target_grid * 2 + target_rewards)
            axs[i, j].axis("off")
    plt.savefig(f"{Path(_dir) / Path(fname)}_preview.pdf")


def gen_sfilling_envs(fname, grid_size=10, N=1000, _dir="gen/data/sfilling/simple"):
    # plot the grid as a black and white heatmap
    agg_grids = []
    agg_pts = []
    interiors = []
    for _ in tqdm.tqdm(range(N)):
        try:
            grid, pts, interior = rand_env_generation(grid_size)
        except IndexError:
            # when the contour is not a closed loop it will throw an error
            continue
        agg_grids.append(grid)
        agg_pts.append(pts)
        interiors.append(interior)
    # save all of the grids and point into a numpy file
    json.dump(interiors, open(f"{Path(_dir) / Path(fname)}_interiors.npy", "w+"))
    np.save(f"{Path(_dir) / Path(fname)}_grids.npy", np.array(agg_grids))
    np.save(f"{Path(_dir) / Path(fname)}_pts.npy", np.array(agg_pts))

    # sample 64 random grids from all of grids and plot them
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            target_grid = agg_grids[i * 8 + j]
            target_pts = agg_pts[i * 8 + j]
            target_grid[target_pts[0][0], target_pts[0][1]] = 0.5
            axs[i, j].imshow(target_grid)
            axs[i, j].axis("off")
    plt.savefig(f"{Path(_dir) / Path(fname)}_preview.pdf")

    # plot the distribution of the number of interior points
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.hist([len(v) for v in interiors], bins=100)
    plt.savefig(f"{Path(_dir) / Path(fname)}_interior_dist.pdf")


def gen_pfinding_envs(
    fname,
    N=1000,
    grid_size=10,
    num_obstacles=7,
    obstacle_size=3,
    _dir="gen/data/pfinding/obs",
):
    grids = []
    for _ in tqdm.tqdm(range(N)):
        grid, start, end = gen_random_pfilling_env(
            grid_size, num=num_obstacles, obstacle_size=obstacle_size
        )
        if end not in get_inside_of_contour(grid, start):
            continue
        grids.append(grid)
    print(len(grids))

    np.save(f"{Path(_dir) / Path(fname)}.npy", np.array(grids))
    # sample 64 random grids from all of grids and plot them
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(grids[i * 5 + j])
            axs[i, j].axis("off")
    plt.savefig(f"{Path(_dir) / Path(fname)}_preview.pdf")
    # save the grids


def gen_no_obstacle_pfinding(
    fname, grid_size=10, N=100, _dir="gen/data/pfinding/simple"
):
    grids = []
    for _ in tqdm.tqdm(range(N)):
        grid, start, end = rand_env_empty_generation(grid_size)
        grids.append(grid)

    np.save(f"{Path(_dir)/Path(fname)}.npy", np.array(grids))
    # sample 64 random grids from all of the grids and plot them
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            target_grid = grids[i * 8 + j]
            axs[i, j].imshow(target_grid)
            axs[i, j].axis("off")
    plt.savefig(f"{Path(_dir) / Path(fname)}_preview.pdf")


def gen_matching_envs(
    fname, grid_size, size=None, N=1000, _dir="gen/data/matching/simple"
):
    grids = []
    for _ in tqdm.tqdm(range(N)):
        grid, start = rand_env_empty_generation(grid_size, output_end_coor=False)
        grids.append(create_one_random_shape(grid, start, size))
    np.save(f"{Path(_dir) / Path(fname)}_grids.npy", np.array(grids))

    # sample 64 random grids from all of the grids and plot them
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            target_grid = grids[i * 8 + j]
            axs[i, j].imshow(target_grid)
            axs[i, j].axis("off")
    plt.savefig(f"{Path(_dir) / Path(fname)}_preview.pdf")


def gen_diff_matching_envs(
    fname, grid_size, size=None, N=1000, _dir="gen/data/matching/diff"
):
    grids = []
    rewards = []
    for _ in tqdm.tqdm(range(N)):
        grid, start = rand_env_empty_generation(grid_size, output_end_coor=False)
        grid_matching = create_one_random_shape(grid, start, size)
        # now create a new grid and fill it with rewards
        reward = grid_matching * (2 * np.random.random(grid.shape) - 1)
        rewards.append(reward)
        grids.append(grid_matching)

    np.save(f"{Path(_dir) / Path(fname)}_grids.npy", np.array(grids))
    np.save(f"{Path(_dir) / Path(fname)}_rewards.npy", np.array(rewards))

    # sample 64 random grids from all of the grids and plot them
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            target_grid = grids[i * 8 + j]
            target_rewards = rewards[i * 8 + j]
            axs[i, j].imshow(target_grid * 2 + target_rewards)
            axs[i, j].axis("off")
    plt.savefig(f"{Path(_dir) / Path(fname)}_preview.pdf")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=[
            "pfinding",
            "obs-pfinding",
            "sfilling",
            "diff-sfilling",
            "matching",
            "diff-matching",
        ],
    )
    parser.add_argument("dir", type=str, help="output directory")
    parser.add_argument("fname", type=str, help="filename of the grids–no extension")
    parser.add_argument("N", type=int, help="the number of environments to generate")
    parser.add_argument(
        "-grid_size", type=int, help="size of the grid being generated", default=8
    )
    parser.add_argument(
        "-num_obstacles",
        type=int,
        help="number of obstables, only applies to obs-pfinding",
    )
    parser.add_argument(
        "-obstacle_size",
        type=int,
        help="size of each obstable, only applies to obs-pfinding",
    )
    parser.add_argument(
        "-matching-size",
        type=int,
        help="size of the shape being mathced, only applies to matching and diff-matching",
    )
    # TODO: add another options that controls the variance of the rewards for the
    # environment. The higher the variance the harder the validation task.
    options = parser.parse_args()
    if options.task == "pfinding":
        gen_no_obstacle_pfinding(
            N=options.N,
            grid_size=options.grid_size,
            fname=options.fname,
            _dir=options.dir,
        )
    elif options.task == "obs-pfinding":
        gen_pfinding_envs(
            N=options.N,
            grid_size=options.grid_size,
            fname=options.fname,
            _dir=options.dir,
            num_obstacles=options.num_obstacles,
            obstacle_size=options.obstacle_size,
        )
    elif options.task == "sfilling":
        gen_sfilling_envs(
            fname=options.fname,
            grid_size=options.grid_size,
            N=options.N,
            _dir=options.dir,
        )
    elif options.task == "diff-sfilling":
        gen_diff_sfilling_envs(
            fname=options.fname,
            grid_size=options.grid_size,
            N=options.N,
            _dir=options.dir,
        )
    elif options.task == "matching":
        gen_matching_envs(
            fname=options.fname,
            grid_size=options.grid_size,
            N=options.N,
            size=options.matching_size,
            _dir=options.dir,
        )
    elif options.task == "diff-matching":
        gen_diff_matching_envs(
            fname=options.fname,
            grid_size=options.grid_size,
            N=options.N,
            size=options.matching_size,
            _dir=options.dir,
        )
