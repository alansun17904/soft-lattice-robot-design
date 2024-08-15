import torch
import argparse
import numpy as np
import utils
from tqdm import tqdm
from transformers import pipeline
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, BartForCausalLM, AutoModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument(
    "model_path", type=str, help="file path of the model to test"
)

def print_grid(grid):
    # Helper function to print the grid neatly
    for row in range(len(grid), 0, -1):
        print(grid[row-1])
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

def expand_grid(sequence, original_size, new_size):
    old_rows, old_cols = original_size, original_size
    new_rows, new_cols = new_size, new_size
    
    # Initialize the new grid with zeros
    new_grid = [[0 for _ in range(new_cols)] for _ in range(new_rows)]
    
    # Copy the original grid values to the new grid
    for i in range(old_rows):
        for j in range(old_cols):
            new_grid[i][j] = sequence[i * old_cols + j]
    
    # Extract the new sequence from the new grid
    new_sequence = []
    for i in range(new_rows):
        for j in range(new_cols):
            new_sequence.append(new_grid[i][j])
    
    return new_sequence


def plot_robot(grid):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Blues', origin='lower')
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.set_xticks(np.arange(-.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(grid), 1), minor=True)
    ax.tick_params(which='both', size=0, labelbottom=False, labelleft=False)
    plt.show()

def text_to_distance(input_text, obj, target_distance):
    print(input_text)
    grid = parse_commands(input_text)
    grid = shrink_grid(grid)
    
    output_sequence = [grid[i][j] for i in range(len(grid)) for j in range(len(grid[0]))]
    print("Output Sequence:", output_sequence)
    #plot_robot(grid)
     
     
    distance, t = utils.calculate_reward(np.array(output_sequence, dtype=np.float32), len(grid), obj, target_distance)
    return output_sequence, distance, t

def check_dataset(sequence):
    
     
    original_size = int(np.sqrt(len(sequence)))
    if (original_size <= 3):
        sequence3 = expand_grid(sequence, original_size, 3)
    if (original_size <= 4):
        sequence4 = expand_grid(sequence, original_size, 4)
    if (original_size <= 5):
        sequence5 = expand_grid(sequence, original_size, 5)
    
    
    f = open("./stable45_1000.txt", "r")
    #print (state)
    for line in f:
        if original_size <= 4:
            if (line.split(",")[0]).replace('.', '').replace(' ',', ') == str(sequence4):
                return True
        if original_size <= 5:
            if (line.split(",")[0]).replace('.', '').replace(' ',', ') == str(sequence5):
                return True


    f = open("./stable45_1000_reverse.txt", "r")
    #print (state)
    for line in f:
        if original_size <= 4:
            if (line.split(",")[0]).replace('.', '').replace(' ',', ') == str(sequence4):
                return True
        if original_size <= 5:
            if (line.split(",")[0]).replace('.', '').replace(' ',', ') == str(sequence5):
                return True
    """
    f = open("./down_45_stairs.txt", "r")
    #print (state)
    for line in f:
        if original_size <= 4:
            if (line.split(",")[0]).replace('.', '').replace(' ',', ') == str(sequence4):
                return True
        if original_size <= 5:
            if (line.split(",")[0]).replace('.', '').replace(' ',', ') == str(sequence5):
                return True
    """
    return False


model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Load the model
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#tokenizer.add_tokens(["<def>", "<add>", "<n>", "<b>", "<e>", "<w>", "|"])



#model = BartForCausalLM.from_pretrained("facebook/bart-base")
#tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

#model = AutoModelForCausalLM.from_pretrained("gpt2")
#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(tokenizer))

model.load_state_dict(torch.load("./all_model/aspect_ratio.pth"))
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


target_distance = 0.5
max_blocks = 10
min_blocks = 5
#prompt = "<|endoftext|> We are designing a soft modular robot for walking from left to right on a plane. Choose the better congfiguration between the following two <|endoftext|>(a)create block0; create block1; create block2; create block3; place block1 to the right of block0; place block2 to the right of block1; place block3 at the bottom of block2<|endoftext|> (b)create block0; create block1; create block2; create block3; create block4; create block5; place block1 at the bottom of block0; place block2 to the right of block0; place block3 at the bottom of block1; place block4 to the right of block3; place block5 to the left of block3<|endoftext|>"

#prompt = "<|endoftext|> We are designing a soft modular robot for walking from left to right on a plane. Choose the better congfiguration between the following two <|endoftext|>(a)create block0; create block1; create block2; create block3; place block1 at the bottom of block0; place block2 to the left of block1; place block3 at the bottom of block2<|endoftext|> (b)create block0; create block1; create block2; create block3; place block1 at the bottom of block0; place block2 at the bottom of block1; place block3 to the left of block2<|endoftext|>"

#prompt = "<|endoftext|> Please design a robot design for walking from left to right on a plane. One of the possible designs is <|endoftext|>(a)create block0; create block1; create block2; create block3; place block1 at the bottom of block0; place block2 to the left of block1; place block3 at the bottom of block2<|endoftext|>. Use at most 25 blocks, and design a better robot of this goal.<|endoftext|>"

prompt = f"<|endoftext|> Please design a robot design for walking from left to right on a plane:<|endoftext|> create block0; create block1;"
#prompt = f"<|endoftext|> Please design a robot design for walking at least {target_distance} distance from left to right on a plane:<|endoftext|> create block0; create block1;"
#prompt = f"<|endoftext|> Please design a robot design for walking from left to right on a plane, using at least {min_blocks} blocks:<|endoftext|> create block0; create block1;"
#prompt = f"<|endoftext|> Please design a robot design for walking from left to right on a plane, using at most {max_blocks} blocks:<|endoftext|> create block0; create block1;"
#prompt = f"<|endoftext|> Please design a robot design for walking at least {target_distance} distance from left to right on a plane, using at most {max_blocks} blocks:<|endoftext|> create block0; create block1;"


inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
print (inputs)
outputs = model.generate(inputs, max_new_tokens=400, do_sample=True, top_k=5, top_p=0.95)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

success_count_numblock = 0
success_count_distance = 0
success_count = 0
non_training_sets = 0
fail = 0

n_generate = 20

distances = []
time_to_goal = []


for i in tqdm(range(n_generate)):
    outputs = model.generate(inputs, max_new_tokens=400, do_sample=True, top_k=5, top_p=0.95)
    output_string = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(output_string)
    print(output_string[0])
    print(type(output_string[0]))
    output_string[0] = "create block0" + output_string[0].split("create block0", 1)[1]
    #output_sequence, distance = text_to_distance(output_string[0], 0)
    #print (distance)

    distance = 0
    #output_sequence, distance, t = text_to_distance(output_string[0], 11, target_distance)
    output_sequence, distance, t = text_to_distance(output_string[0], 12, target_distance)
    print (output_sequence)
    if (distance == -1  and t == -1):
        fail = fail + 1
        continue

    num_blocks = output_sequence.count(1)
    
    if num_blocks >= min_blocks: #<= max_blocks:
        success_count_numblock = success_count_numblock + 1
    
    if distance > target_distance:
        success_count_distance = success_count_distance + 1

    #if num_blocks >= min_blocks and distance > target_distance:
    if distance > target_distance:
        success_count = success_count + 1
        time_to_goal.append(t)

    if not check_dataset(output_sequence):
        non_training_sets = non_training_sets + 1



    distances.append(distance)

    


print ("Metric 1", success_count/n_generate)
print (success_count_numblock/n_generate)
print (success_count_distance/n_generate)
print ("Metric 2", np.mean(distances), np.var(distances))
print ("Metric 3", np.mean(time_to_goal), np.var(time_to_goal))
print ("Metric 4", non_training_sets/n_generate)
print ("Metric 5", fail/n_generate)
