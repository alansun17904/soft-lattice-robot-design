# Soft-Lattice Robot 2D Simulations
This directory handles all of the simulations for the soft-lattice robots which can either be used for inference and validation of the trained language models or it can be used to generate expert demonstrations for the language models. Here is the general workflow:

1. Generate the soft-lattice robot designs and format them into `NxN` binary matrices where 1 denotes that there should be a robot mini there and 0 denotes the absence of a robot mini. Format all of these matrices into a `json` file as a list of `NxN` binary matrices.

2. Using this json (as a blueprint), generate the configurations of the robots as they are to be built in `difftaichi`. To do this simply run the Python script:
```python
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
- Flat but with one variable gap size
- Flat but robot starts on the other side of the map