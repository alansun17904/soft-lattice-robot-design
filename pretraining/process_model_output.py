import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "input_file",
    type=str,
    help="file path of file that contains the robot string",
)
options = parser.parse_args()

def print_grid(grid):
    # Helper function to print the grid neatly
    for row in grid:
        print(row)
    print()  # Blank line for better separation

def parse_commands(input_text):
    commands = input_text.split("; ")
#     print(commands)
    blocks = {}
    grid_size = 10  # Start with a 5x5 grid
    grid = [[0] * grid_size for _ in range(grid_size)]
    middle = grid_size // 2  # Middle point in the grid

    for command in commands:
        if "create" in command:
            block_id = command.split(" ")[1]
            if not blocks:  # If this is the first block
                blocks[block_id] = (middle, middle)  # Place the first block in the middle
                grid[middle][middle] = 1  # Mark the middle of the grid
#                 print("After creating", block_id, "at the middle:")
#                 print_grid(grid)
            else:
                blocks[block_id] = (None, None)  # Others don't have an initial position yet
        elif "place" in command:
#             print("the cur command: ", command)
            parts = command.split(" ")
            block_to_place = parts[1]
            relative_position = parts[4:]
            reference_block = parts[-1].strip(";")
            if reference_block in blocks and blocks[reference_block] != (None, None):
                ref_x, ref_y = blocks[reference_block]
                new_x, new_y = ref_x, ref_y  # Default to current position
                if "left" in relative_position:
                    new_x = ref_x - 1
                elif "right" in relative_position:
                    new_x = ref_x + 1
                elif "top" in relative_position:
                    new_y = ref_y + 1
                elif "bottom" in relative_position:
                    new_y = ref_y - 1
                
                # Update the block's position if within grid bounds
                if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                    blocks[block_to_place] = (new_x, new_y)
                    grid[new_y][new_x] = 1
#                     print_grid(grid)
    
    return grid

    

def shrink_grid(grid):
    grid = np.array(grid)
    non_empty_columns = np.where(grid.any(axis=0))[0]
    non_empty_rows = np.where(grid.any(axis=1))[0]
    grid = grid[np.ix_(non_empty_rows, non_empty_columns)]

    # Determine the number of rows and columns
    rows, columns = grid.shape

    # Add rows or columns based on the difference
    if rows != columns:
        difference = abs(rows - columns)
        if rows > columns:
            # Add columns to the right
            new_grid = np.zeros((rows, rows), dtype=grid.dtype)  # new grid with equal dimensions
            new_grid[:rows, :columns] = grid
        else:
            # Add rows to the top
            new_grid = np.zeros((columns, columns), dtype=grid.dtype)  # new grid with equal dimensions
            new_grid[:rows, :columns] = grid

        grid = new_grid

    print("Final Grid:")
    print_grid(grid)
    return grid

def plot_robot(grid):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Blues', origin='lower')
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.set_xticks(np.arange(-.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(grid), 1), minor=True)
    ax.tick_params(which='both', size=0, labelbottom=False, labelleft=False)
    plt.show()

def main():
    
    f = open(options.input_file)
    input_text = f.readline()
    f.close()

    print(input_text)
    grid = parse_commands(input_text)
    grid = shrink_grid(grid)
    
    output_sequence = [grid[i][j] for i in range(len(grid)) for j in range(len(grid[0]))]
    print("Output Sequence:", output_sequence)
    plot_robot(grid)
     
    
    utils.calculate_reward(np.array(output_sequence, dtype=np.float32))

if __name__ == "__main__":
    main()