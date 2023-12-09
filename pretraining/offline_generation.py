# Randomly generate trajectories for offline training
import json
import pickle
import random
import numpy as np
from tabulate import tabulate


############################
# Constants
############################
N = 100000  # number of trajectories
ROBOT_SIDE = 3
ROBOT_AREA = 9  # 3 x 3 grid
T = ROBOT_AREA * 2  # number of steps in each trajectory
MODE = [0, 1, 2]  # 0: add, 1: delete, 2: pass
CONFIGS = json.load(open("two_simulation/valid_robots.json", "r"))
CONFIGS = [np.array(v) for v in CONFIGS]  # convert all configs to np.arrays
SCORES = json.load(open("two_simulation/robot_scores.json", "r"))
HASHED = json.load(open("data/3x3-t-invariant-mapping.json", "r"))


############################
# Saved Data
############################
trajectories = []


############################
# Definitions
# State: Robot configuration described as an integer which corresponds to
# the robot in the valid_robots.json file
# Action: (mode, target) where mode is 0 for add, 1 for delete, 2 for pass
# and target is the index of the target to add/delete
############################


def valid_removal(state):
    a = CONFIGS[state]
    if np.sum(a) == 1:
        return []
    return np.nonzero(a)[0]


def valid_addition(state):
    a = CONFIGS[state].reshape(ROBOT_SIDE, ROBOT_SIDE)
    valid = []
    for i in range(ROBOT_AREA):
        r, c = i // ROBOT_SIDE, i % ROBOT_SIDE
        if a[r, c] == 0:
            if r > 0 and a[r - 1, c] == 1:
                valid.append(i)
            elif r < len(a) - 1 and a[r + 1, c] == 1:
                valid.append(i)
            elif c > 0 and a[r, c - 1] == 1:
                valid.append(i)
            elif c < len(a[0]) - 1 and a[r, c + 1] == 1:
                valid.append(i)
    return valid


def get_next_state(state, action):
    mode, target = action
    if mode == 2:
        return state
    curr = CONFIGS[state]
    curr[target] = 1 if mode == 0 else 0


def generate_next_state(state, mode):
    """
    Given the current state (as a robot), generate
    the next state given the mode.
    """
    # get the robot associated with the current state
    robot = CONFIGS[state].copy()
    next_ = None
    if mode == 0:
        valid_additions = valid_addition(state)
        if len(valid_additions) == 0:
            return state, next_
        else:
            next_ = np.random.choice(valid_additions)
            robot[next_] = 1
    elif mode == 1:
        valid_removals = valid_removal(state)
        if len(valid_removals) == 0:
            return state, next_
        else:
            next_ = np.random.choice(valid_removals)
            robot[next_] = 0
    else:
        return state, next_
    # get the next state based on the hashed string
    hashed = "".join([str(int(x)) for x in robot])
    if hashed not in HASHED:
        return state, None
    return (HASHED[hashed], None if next_ == -1 else next_)


def total_reward(traj):
    # check if the trajectory is valid
    if traj[-1][1][0] != 2:
        return -1000
    # get the score of the final robot
    score = SCORES[str(traj[-1][0])]
    # cross entropy loss
    dist = np.log(1.2 + score + 1e-8)  # the score is negative, we want minimize it
    robot_size = np.sum(CONFIGS[traj[-1][0]])
    return dist * robot_size - (1 / 9) * len(traj)


if __name__ == "__main__":
    for j in range(N):
        trajectory = []
        curr_state = 0
        t = 1
        while True:
            add_prob = 0.95 / t ** (1 / 9)
            probs = [add_prob, 0.05, 0.95 - add_prob]
            if j == 0:
                pass
                # print(probs)
            mode = np.random.choice(MODE, p=probs)
            next_state, next_ = generate_next_state(curr_state, mode)
            # add the current state, action
            # (which we define as the mode and the target index)
            if np.sum(CONFIGS[next_state] ^ CONFIGS[curr_state]) == 0:
                if mode != 2:
                    trajectory.append((curr_state, (mode, next_)))
                    break
                else:
                    trajectory.append((curr_state, (2, 0)))
                    break
            index = np.nonzero(CONFIGS[next_state] ^ CONFIGS[curr_state])[0][0]
            trajectory.append((curr_state, (mode, index)))
            curr_state = next_state
            if mode == 2:
                break
            t += 1
        trajectories.append(trajectory)
    table = (
        ("Trajectories", len(trajectories)),
        ("Avg length", np.mean([len(t) for t in trajectories])),
        (
            "Avg score",
            np.mean(([SCORES[str(traj[-1][0])] for traj in trajectories])),
        ),
        (
            "Avg additions",
            np.mean([np.sum([t[1][0] == 0 for t in traj]) for traj in trajectories]),
        ),
        (
            "Avg deletions",
            np.mean([np.sum([t[1][0] == 1 for t in traj]) for traj in trajectories]),
        ),
        (
            "Avg time",
            np.mean([len(traj) for traj in trajectories if traj[-1][1][0] == 2]),
        ),
        (
            "Avg valid trajectories",
            N - np.sum([1 for traj in trajectories if traj[-1][1][0] != 2]),
        ),
        (
            "Avg total rewards",
            np.mean([total_reward(traj) for traj in trajectories]),
        ),
        (
            "Avg valid rewards",
            np.mean(
                [total_reward(traj) for traj in trajectories if traj[-1][1][0] == 2]
            ),
        ),
    )
    valid_trajectories = [traj for traj in trajectories if traj[-1][1][0] == 2]
    print(tabulate(table, tablefmt="simple_grid"))
    pickle.dump(trajectories, open("data/trajectories.pkl", "wb"))
    pickle.dump(valid_trajectories, open("data/valid_trajectories.pkl", "wb"))
