# Pretraining
This directory handles all of the pretraining for the soft-lattice robots. Here is the general workflow:

1. Using the robot designs and simulation results that can be obtained by running the scripts in `simulation/`, generate all of the GSL programs corresponding to these robots. See the subsection on `GSL` for the syntax of this language. To do this, run the following command

```bash
python3 generate_pretrain_data.py <robot.json> <losses.json> <top-p> <output_file>
```

This will output a `json` file containing a set of GSL programs describing robots given in `robot.json`. Note that the program will only generate programs for the robots that have a loss in the top `p` percentile of all of the robots. The resulting programs will be exported to an `output_file`

2. Pretrain a language model using a causal language modeling objective on these GSL programs. 


## Generation Standard Language (GSL) 
The generation standard language allows generation of robot structures using simple commands (see below). Every program starts with 
the line `def 0` specifying that the block 0 has been placed. 
- `def <name>`: specifies a new robot block. Note that the first block will always be given so an empty program corresponds to just "giving up." Also note that specifying a new block does not actually mean putting it down. 
- `add <source> <target> <direction> <direction>`: adds the `<target>` block to the `<source>` block in a particular direction (specified by `<n>, <s>, <e>, <w>`: north, south, east, west). Note that the directions are always specific to the source block. It should be noted that the source blocks must have already been placed. The second direction corresponds to the direction of the target block. Note that this is only used for aggregate structures. 
- `remove <source>`
- `aggregate <name>, <n>, <s>, <e>, <w>: <name1> <name2> <name3> ...`: creates a structure that is composed of `<name1>, <name2>, ...`. This can be then called like a function later on. The parameters `<n>, <s>, <w>, <e>` correspond to the blocks that will take on the north, south, east, west of this aggregate structure. This will be the target of the `add` command. 

See below for a sample program:
```
def 0
def 1
def 2
def 3
def 4
def 5
def 6

add 0 1 <n>
add 1 2 <n>
add 2 3 <e>
add 3 4 <e>
add 4 5 <s>
add 5 6 <s>
```
This program generates the following robot:
```
 _______  _______  _______
|       ||       ||       |
|   2   ||   3   ||   4   |
|_______||_______||_______|   
 _______           _______
|       |         |       |
|   1   |         |   5   |
|_______|         |_______|
 _______           _______
|       |         |       |
|   0   |         |   6   |
|_______|         |_______| 
```

