# Soft-Lattice Robot 2D Simulations
This directory handles all of the simulations for the soft-lattice robots which can either be used for inference and validation of the trained language models or it can be used to generate expert demonstrations for the language models. Here is the general workflow:

1. Generate all robots (i.e. all valid binary matrices). Run the following:
```bash
python3 generate_all_robots.py <n> <m>
```
Here, `<n>` corresponds to the maximum number of blocks that can be used and `<m>` refers to the size of the binary matrix (i.e. the size of the constrained environment). It must be that `n <= m**2`, otherwise the script will throw an error.

<!-- 1. Generate the soft-lattice robot designs and format them into `NxN` binary matrices where 1 denotes that there should be a robot mini there and 0 denotes the absence of a robot mini. Format all of these matrices into a `json` file as a list of `NxN` binary matrices. -->

2. Using this json (as a blueprint), generate the configurations of the robots as they are to be built in `difftaichi`. To do this simply run the Python script:
```bash
python3 generate_robot_config.py <json-file-of-robots> robots/
```
This will then output a series of files in the format of `<i>.json` where `i` is the index of the robot with respect to the original list of robots. After generating these object/spring/angle-spring configurations the simulation can now be run. 

3. Simply run the script
```bash
./simulate.sh
```
This will simulate all of the robots in the `robots/` directory and export their losses over each optimization iteration in the file `losses.json`. 

## Example 
```bash
python3 generate_robot_config.py sample_robot.json robots/
./simulate.sh
```

## More Complicated Environments
In this section, we cover how to enable more complicated environments. Currently supported:
- Staircase
    - The number of stairs (note that if there are no stairs, i.e. only flat ground, then we say that there is only one stair)
    - The widths of each stair 
    - The height of each stair 
    - x,y coordinate of the goal, defaults to `[0.9, 0.2]`
- Flat but with one variable gap size
    - Need to pass the number of gaps as a positive integer
    - The starting x-coordinate of the gaps as a float between `[0,1]`
    - The width of the respective gaps as a float between `[0,1]`
- Flat but robot starts on the other side of the map
    - Do not need to pass any additional arguments.
Below are the changes in the actual code that I made to achieve these for reference:

| Technique | Notes |
|---|---|
| Reverse | - Changed the goal from `[0.9,0.2]` to `[0.1,0.2]` and inverted the sign on the loss function. |
| Gap | - Added three two arguments to the script `gaps`, `gap_starts`, `gap_width`. They represent<br>the total number of gaps, the `x`-coordinate of where the gaps start, and the coordinate-width<br>of each gap.<br>- Added GUI support for these gaps, see the `forward` function.<br>- Added "falling" detection in `advance_toi` and `advance_no_toi` for each object. Subroutine<br>flags true when the object is above the gap. Then, it is accelerated downwards at<br>`gravity * (dt) ** 2` (gravity is already halved).<br>- Changed existing ground velocity mechanism in `advance_toi` and `advance_no_toi` to only<br>activate when above the gap.<br>**Existing Issues:** Numerically unstable when robot falls into the gap, dives sideways and all the gradients go to `nan`.|
| Staircase |  - Added three arguments to the script `stairs`, `stair_widths`, and `stair_heights`. They represent the total number of stairs, the width of each stairs (a number between 0-1), and the height of each stairs.<br>-Modified `advance_toi` and `advance_no_toi` according to Matthew's documentation for each object.<br>**Existing Issues:**<br>Numerical instability when the robot reaches the end of one stair and the start of another stair. Not too sure what is happening here.|

Some current issues that need to be resolved:
1. Numerical instability of simulation with gaps.
2. Numerical instability of simulation with stairs.