"""
Given some N, generates all robots that fit into an NxN matrix note that the
robots are translation invariant. 
"""


import json
import argparse
import numpy as np


class RobotSeq:
    def __init__(self, seq, max_dim):
        self.dim = max_dim
        self.seq = tuple(seq)

        # normalize the sequence
        r = self.make_robot()
        r = RobotSeq.normalize_shape(r)
        self.seq = tuple(zip(*r.nonzero()))

    def make_robot(self):
        robot = np.zeros((self.dim, self.dim))
        for step in self.seq:
            robot[step] = 1
        return robot

    def __eq__(self, other):
        # convert both robots to matrices
        return self.seq == other.seq

    def __hash__(self) -> int:
        return hash(self.seq)

    @staticmethod
    def normalize_shape(matrix):
        """Given a binary matrix, move the one entries to the top right corner"""
        while np.all(matrix[:, 0] == 0):  # shift by one column
            matrix = np.roll(matrix, -1, axis=1)
        while np.all(matrix[0, :] == 0):
            matrix = np.roll(matrix, -1, axis=0)
        return matrix

    def extend_robot_by_one(self):
        # make the robot
        robot = self.make_robot()
        xs, ys = np.nonzero(robot)
        new_robots = set()
        for i in range(len(xs)):
            x, y = xs[i], ys[i]
            for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                i, j = (
                    min(max(x + dx, 0), len(robot) - 1),
                    min(max(y + dy, 0), len(robot) - 1),
                )
                if robot[i, j] == 0:
                    new_robot = RobotSeq(list(self.seq) + [(i, j)], len(robot))
                    new_robots.add(new_robot)
        return new_robots

    def __str__(self):
        r = self.make_robot()
        out = ""
        for v in r:
            out += " ".join(map(lambda x: str(int(x)), v)) + "\n"
        return out


def n_block_robots(n, max_dim=5):
    """Generates all possible robots that have `<= n` blocks which
    fit in an `max_dim x max_dim` matrix. Note that it must be that
    `max_dim**2 >= n`. Note that we are returning a build sequence.
    """
    dp = [
        set((RobotSeq(((0, 0),), max_dim),)),
    ]
    for i in range(n - 1):
        robots = set()
        for robot in dp[i]:
            new_robots = robot.extend_robot_by_one()
            for new_robot in new_robots:
                robots.add(new_robot)
        dp.append(robots)

    return dp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="the total number of blocks")
    parser.add_argument("m", type=int, help="maximum side length of matrix")
    parser.add_argument("output_file", type=str, help="output json file")
    options = parser.parse_args()

    if options.n > options.m**2:
        assert RuntimeError(
            "must be that n <= m**2 otherwise there is no space to create robots"
        )

    robots = n_block_robots(options.n, options.m)
    # print all of the robots

    all_robots = []
    for i, size in enumerate(robots):
        print(f"All Robots with {i+1} Blocks\n{'-'*40}")
        for robot in size:
            all_robots.append(robot.make_robot().astype(int).tolist())
            print(robot)
    print(all_robots)
    json.dump(all_robots, open(options.output_file, "w+"))
