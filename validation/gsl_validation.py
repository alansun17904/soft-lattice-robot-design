import re


direction_map = {"n": "s", "w": "e", "s": "n", "e": "w"}
direction_coords = {"n": (0, 1), "s": (0, -1), "e": (1, 0), "w": (-1, 0)}


def valid_gsl(s) -> bool:
    """checks if the string input is a valid gsl"""
    lines = s.strip().split("\n")
    definitions = set()
    # absolute coordinate for each block
    obj_coords = {}
    # dict mapping object to a dict mapping direction to object_id
    obj_pairs = {}

    for line in lines:
        parts = line.split()
        parts = list(filter(lambda x: x.strip(), parts))
        if not parts:
            continue
        if parts[0] == "def":
            if not creation_command(parts, definitions):
                return False
        elif parts[0] == "add":
            if not add_command(parts, obj_coords, obj_pairs, definitions):
                return False
        elif parts[0] == "remove":
            if not remove_command(parts, obj_coords, obj_pairs, definitions):
                return False
        else:
            return False
    return True


def creation_command(parts, definitions) -> bool:
    """checks existence of object and stores it in a set"""

    if len(parts) != 2:
        return False
    definition = parts[1]
    if definition in definitions:
        return False
    definitions.add(definition)
    return True


def add_command(parts, obj_coords, obj_pairs, definitions) -> bool:
    """adds a defined object to the structure

    Parameters:
    parts -- the line which includes the current command
    obj_coords -- hashtable mapping and object to its coordinates
    obj_pairs -- hastable mapping and object to a hashtable maping a direction to a destination
    definitions -- a set containing all the defined objects
    """

    if len(parts) != 4:
        return False
    source = parts[1]
    destination = parts[2]
    direction = re.match("<([n,w,s,e])>", parts[3])
    if direction is None:
        return False
    else:
        direction = direction.groups()[0]

    if source not in definitions or destination not in definitions:
        return False

    if len(obj_coords) == 0:
        obj_coords[source] = (0, 0)
        obj_pairs[source] = dict()
        obj_pairs[source][direction] = destination

        obj_coords[destination] = direction_coords[direction_map[direction]]
        obj_pairs[destination] = dict()
        obj_pairs[destination][direction_map[direction]] = source

    elif destination in obj_coords:
        if not check_src_dest_compatible(obj_pairs, direction, source, destination):
            return False
        obj_pairs[source][
            direction
        ] = destination  # mapping direction of source to destination

        source_x = obj_coords[source][0]  # the x-coordinate of source
        source_y = obj_coords[source][1]  # the y-coordinate of source
        destination_x = obj_coords[destination][0]  # the x-coordinate of destination
        destination_y = obj_coords[destination][1]  # the y-coordinate of destination

        # if statements check whether the coordinates of the destination object will be compatible with the source in that direction
        if direction == "n" or direction == "s":
            if (source_x != destination_x) or (
                source_y - destination_y != (1 if direction == "n" else -1)
            ):
                return False
        else:
            if (source_x - destination_x != (-1 if direction == "e" else 1)) or (
                source_y != destination_y
            ):
                return False
        if not check_src_dest_compatible(obj_pairs, direction, source, destination):
            return False
        obj_pairs[destination][direction] = source
    else:
        if not check_src_dest_compatible(obj_pairs, direction, source, destination):
            return False
        set_add(obj_pairs, direction, source, destination, obj_coords)
    return True


def check_src_dest_compatible(obj_pairs, direction, source, destination) -> bool:
    """checks if destination can be added to source in the specified direction"""
    if direction in obj_pairs[source] and destination in obj_pairs:
        if direction_map[direction] in obj_pairs[destination]:
            if obj_pairs[source][direction] != destination:
                return False
    if direction in obj_pairs[source] and destination not in obj_pairs:
        return False
    return True


def set_add(obj_pairs, direction, source, destination, obj_coords) -> bool:
    """defines the connections between source and destination"""

    obj_pairs[source][direction] = destination
    obj_pairs[destination] = dict()
    obj_pairs[destination][direction_map[direction]] = source
    if direction == "n":
        obj_coords[destination] = (obj_coords[source][0], obj_coords[source][1] - 1)
    elif direction == "s":
        obj_coords[destination] = (obj_coords[source][0], obj_coords[source][1] + 1)
    elif direction == "e":
        obj_coords[destination] = (obj_coords[source][0] + 1, obj_coords[source][1])
    else:
        obj_coords[destination] = (obj_coords[source][0] - 1, obj_coords[source][1])


def remove_command(parts, obj_coords, obj_pairs, definitions) -> bool:
    """removes object from structure"""

    if len(parts) != 2 or parts[1] not in definitions:
        return False
    else:
        obj_coords.pop(parts[1])  # remove object from hashtable with coordinates
        for i in obj_pairs:
            for j in obj_pairs[i]:
                if obj_pairs[i][j] == parts[1]:
                    obj_pairs[i].pop(
                        j
                    )  # remove any direction in which the object is connected
                    break
        obj_pairs.pop(parts[1])  # remove object from the map of maps
    return True
