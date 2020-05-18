# Learning

As the name indicates, the Learning directory contains the code for learning control policies to automate 
polyculture farming. This README serves as a high-level documentation of the class files.

## Files structure

The most important (class) files are mentioned in this table.

| **Function**               | **Files and classes**                                                                 | **Description**               |
|----------------------------| ------------------------------------------------------------------------------------- |-----------------------------|
| Data generation            |  [collect.py](AlphaGarden/Learning/collect.py) <br><br> [data_collection.py](AlphaGarden/Learning/data_collection.py) <br><br> [split.py](AlphaGarden/Learning/split.py) <br><br> [moments.py](AlphaGarden/Learning/moments.py)                  | data_collection.py can be used to generate images and state data for the garden.                      |
| Supervised Learning        |  [constants.py](AlphaGarden/Learning/constants.py) <br><br> [dataset.py](AlphaGarden/Learning/dataset.py) <br><br> [eval_policy.py](AlphaGarden/Learning/eval_policy.py)                                                                         |eval_policy can be used to evaluate the existing policies. |
| Reinforcement Learning     |  [cnn_policy.py](AlphaGarden/Learning/cnn_policy.py) <br><br> [file_utils.py](AlphaGarden/Learning/file_utils.py) <br><br> [graph_utils.py](AlphaGarden/Learning/graph_utils.py) <br><br> [rl_pipeline.py](AlphaGarden/Learning/rl_pipeline.py)  |rl_pipeline.py is deprecated. Containing code for reinforcement learning policy search. |
|         ...                |                  ...                                                                                                                                                                                                                             | ... |
