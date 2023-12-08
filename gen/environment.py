import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt


def rand_env_empty_generation(n: int) -> np.array:
    start = (random.randint(0, n - 1), random.randint(0, n - 1))
    end = (random.randint(0, n - 1), random.randint(0, n - 1))
    grid = np.zeros((n, n))
    grid[start] = 0.5
    grid[end] = 0.5
    return grid, start, end


def rand_env_generation_diff_rewards(n: int, pts: int = 1) -> np.array:
    contour, start, interior = rand_env_generation(n, pts)
    rewards = [2 * random.random() - 1 for _ in range(len(interior))]
    reward_grid = np.zeros((n, n))
    for (x, y), r in zip(interior, rewards):
        reward_grid[x, y] = r
    return contour, start, interior, reward_grid


def rand_env_generation(n: int, pts: int = 1) -> np.array:
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


def gen_diff_sfilling_envs(N=1000, fname="gen/data/sfilling/diff"):
    agg_grids = []
    agg_pts = []
    rewards = []
    for _ in tqdm.tqdm(range(N)):
        try:
            grid, pts, _, rewrads = rand_env_generation_diff_rewards(16)
        except IndexError:
            continue
        agg_grids.append(grid)
        agg_pts.append(pts)
        rewards.append(rewrads)
    np.save(f"{fname}/grids.npy", np.array(agg_grids))
    np.save(f"{fname}/pts.npy", np.array(agg_pts))
    np.save(f"{fname}/rewards.npy", np.array(rewards))

    # sample 64 random grids from all of the grids and plot them
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            target_grid = agg_grids[i * 8 + j]
            target_rewards = rewards[i * 8 + j]
            axs[i, j].imshow(target_grid * 2 + target_rewards)
            axs[i, j].axis("off")
    plt.savefig(f"{fname}/sample_grids.pdf")


def gen_sfilling_envs(N=1000, fname="gen/data/sfilling/simple"):
    # plot the grid as a black and white heatmap
    agg_grids = []
    agg_pts = []
    interiors = []
    for _ in tqdm.tqdm(range(N)):
        try:
            grid, pts, interior = rand_env_generation(16)
        except IndexError:
            # when the contour is not a closed loop it will throw an error
            continue
        agg_grids.append(grid)
        agg_pts.append(pts)
        interiors.append(interior)
    # save all of the grids and point into a numpy file
    np.save(f"{fname}/test_interiors.npy", np.array(interiors))
    np.save(f"{fname}/test_grids.npy", np.array(agg_grids))
    np.save(f"{fname}/test_pts.npy", np.array(agg_pts))

    # sample 64 random grids from all of grids and plot them
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            target_grid = agg_grids[i * 8 + j]
            target_pts = agg_pts[i * 8 + j]
            target_grid[target_pts[0][0], target_pts[0][1]] = 0.5
            axs[i, j].imshow(target_grid)
            axs[i, j].axis("off")
    plt.savefig(f"{fname}/sample_grids_test.pdf")

    # plot the distribution of the number of interior points
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.hist(interiors, bins=100)
    plt.savefig(f"{fname}/interior_num_dist_test.pdf")


def gen_pfinding_envs(
    N=1000, grid_size=10, num_obstacles=7, obstacle_size=3, fname="gen/data/pfinding"
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

    np.save(f"{fname}/grids.npy", np.array(grids))
    # sample 64 random grids from all of grids and plot them
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(grids[i * 5 + j])
            axs[i, j].axis("off")
    plt.savefig(f"{fname}/sample_grids.pdf")
    # save the grids


def gen_no_obstacle_pfinding(N=100, fname="gen/data/pfinding"):
    grids = []
    for _ in tqdm.tqdm(range(N)):
        grid, start, end = rand_env_empty_generation(10)
        grids.append(grid)
    np.save(f"{fname}/grids_no_obstacle.npy", np.array(grids))


if __name__ == "__main__":
    gen_pfinding_envs(N=1000)
    # gen_diff_sfilling_envs(N=1000000)
