import random
import numpy as np


def gen_matching(environment, start, n, reward=None):
    programs = []
    for _ in range(n):
        seq = get_visit_sequence(environment, start, reward)
        programs.append(generate_gsl_program(seq))
    return programs


def get_visit_sequence(environment, start, reward=None):
    """Given the environment, get a random sequence of points
    that visits all of the target 
    """


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