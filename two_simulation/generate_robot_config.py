""""
Given a list of square, binary matrices. This script converts a json file that represents
robots readable by difftaichi. It is required that for any nxn matrix it must be that
the 1 entries are adjacent. 

The script takes as argument the path to a json file containing a list of these 
nxn binary matrices as well as the output path of where the resulting configurations go.

python generate_robot_config.py <path-to-json-file> <output-file-name>

The resulting json file will be a list of dictionaries that follow the format:
[
    {
        "objects": [],  # a list of coordinates of the robot connection points
        "springs": [],
            # a list of [
            #   index of first connector,
            #   index of second connector,
            #   length,
            #   actuation
            # ]
        "angle_springs": [],
            # a list of [
            #   index of the first connector,
            #   index of the second connector,
            #   index of the third connector,
            #   stiffness (default=1)
            # ]
    }
]

"""

import sys
import json
import uuid


class Robot:
    def __init__(self, matrix, id_len=5):
        """Uses an nxn binary matrix to generate a robot.
        Define the bottom left corner of the matrix as 0,0 and the top-right corner
        as n-1,n-1
        """
        self.objects = []
        self.springs = []
        self.angle_springs = []
        self.name = uuid.uuid4().__str__()[:id_len]

        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 1:
                    self.add_mesh_square(j, n-i-1)

    def add_spring(self, a, b, length=None, stiffness=1.4e4, actuation=0.15):
        """Adds a spring between two objects `a` and `b`.
        :param a: the index of the first connector object from self.objects
        :param b: the index of the second connector object from self.objects
        :param length: length of the string, real number
        :param stiffness: stiffness of string, real number
        :param actuation: +/-% that the spring is able to extend/contract its length
        """
        if length is None:
            length = (
                (self.objects[b][0] - self.objects[a][0]) ** 2
                + (self.objects[b][1] - self.objects[a][1]) ** 2
            ) ** 0.5
        self.springs.append([a, b, length, stiffness, actuation])

    def add_mesh_point(self, i, j):
        """Helper function for adding a new object, checks beforehand if the
        object already exists. If it does, then return its index. Otherwise,
        add the object to the list and return its new index.
        """
        x, y = i * 0.05 + 0.1, j * 0.05 + 0.1
        # check if already an object
        for index, obj in enumerate(self.objects):
            if obj == [x, y]:
                return index
        self.objects.append([x, y])
        return len(self.objects) - 1

    def add_mesh_spring(self, a, b, stiffness, actuation):
        """Helper function for adding a new spring, checks beforehand if the
        spring already exists. If it does, then return its index. Otherwise,
        add the new spring to the list of springs and return its index.
        """
        for index, spring in enumerate(self.springs):
            if spring[:2] == [a, b] or spring[:2] == [b, a]:
                return index
        self.add_spring(a, b, stiffness=stiffness, actuation=actuation)
        return len(self.springs) - 1

    def add_mesh_square(self, x, y, actuation=0.15):
        """Adds a square mini to the robot. The x-y coordinate
        defines the lower-left connect of the square mini (a). Then, 5
        connectors are defined with springs between them as such:

            b ----- d
            | \   / |
            |   e   |
            | /   \ |
            a ----- c

        """
        s = 3e4
        a = self.add_mesh_point(x, y)
        b = self.add_mesh_point(x, y + 1)
        c = self.add_mesh_point(x + 1, y)
        d = self.add_mesh_point(x + 1, y + 1)
        e = self.add_mesh_point(x + 0.5, y + 0.5)
        self.add_mesh_spring(a, b, s, actuation)
        self.add_mesh_spring(c, d, s, actuation)
        self.add_mesh_spring(b, d, s, actuation)
        self.add_mesh_spring(a, c, s, actuation)
        self.add_mesh_spring(a, e, s, 0)
        self.add_mesh_spring(b, e, s, 0)
        self.add_mesh_spring(c, e, s, 0)
        self.add_mesh_spring(d, e, s, 0)
        self.angle_springs.extend(
            [
                [a, b, e, 1],
                [b, c, e, 1],
                [c, d, e, 1],
                [c, a, e, 1],
            ]
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python generate_robot_config.py <path-to-json-file> <output-file-name>"
        )
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    robot_configs = []

    # load the input json file
    matrices = json.load(open(input_filename, "r"))
    for matrix in matrices:
        r = Robot(matrix)
        print(r.name)
        robot_configs.append(
            {
                # "name": r.name,
                "objects": r.objects,
                "springs": r.springs,
                "angle_springs": r.angle_springs,
            }
        )

    json.dump(robot_configs, open(output_filename, "w+"))
