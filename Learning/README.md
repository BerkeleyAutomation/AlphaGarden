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

## Examples

Install the required pip packages and the mentioned packages from the README in the main folder.

### data_collection.py

To run the simulator and collect data:

1. Move to the Learning folder with `cd Learning/`
2. Run `python data_collection.py`

### eval_policy.py

To evaluate automation policies with the simulator:

1. Move to the Learning folder with `cd Learning/`
2. Run `python eval_ploicy.py` to evaluate the the baseline policy with serial execution.

Other command-line options for `eval_ploicy.py` are:
* `'-t', '--tests', type=int, default=1` -- Number of evaulation trials.
* `'-n', '--net', type=str, default='/'` -- To evaluate a learned policy supply the trained params.
* `'-m', '--moments', type=str, default='/'` -- To evaluate a learned policy supply the moments of the dataset used to train the policy.
* `'-s', '--seed', type=int, default=0` -- Numpy's random seed.
* `'-p', '--policy', type=str, default='b'` -- baseline [b], naive baseline [n], learned [l], irrigation [i].
* `'--multi', action='store_true'` -- Enable multiprocessing.
* `'-l', '--threshold', type=float, default=1.0` -- Prune threshold
* `'-d', '--days', type=int, default=100` -- Garden days
