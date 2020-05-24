# Learning

As the name indicates, the Learning directory contains the code for training and evaluation of control policies to automate 
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

To train a supervised learning policy we need training data. Experiments showed that running 90-150 gardens with 200 plants from 10 plant types, 
0.1 _PERCENT_NON_PLANT_CENTERS_ and 100 garden days deliver a reasonable amount of training data. 

We can generate the data for one garden with `data_collection.py` or for a batch with `collect.py`.
After having generating data, next step is to normalize the generated data with `moments.py`.
To adjust the network architecture update the code in `net.py`. Before training, additional adjustments may be needed to be made in `trainer.py`, e.g. set GPU usage or amount of workers for loading training data.
Before training, the normalised data generated with `moments.py` needs to be connected to the generated data. Last but not least, training is happening with `train.py`.

You may want to run this in a container. There is a [Dockerfile](AlphaGarden/Dockerfile) provided in the root folder.

In the following we discribe to pipeline in detail. 

#### data_collection.py

To run the simulator and collect data for a garden:

1. Move to the Learning folder with `cd Learning/`
2. Run `python data_collection.py`

#### collect.py

To run the simulator and collect data in a batch:

1. Move to the Learning folder with `cd Learning/` 
2. Run `python collect.py -d YOUR_DIRECTORY -n NUMBER_OF_BATCHES` where you specify the output directory and number of batches. 
**Note**: 150 gardens may generate over 0.5 TB of data and the maximal amount of files per directory might be exceeded.

#### moments.py

Update path to data directory. Code needs to be modified if data is spread across several directories. 

#### net.py

Adjust the network architecture if needed.

Line 33-39 are the conv layers for png images (variables start with *cc_*). <br>
Line 41-47 are the layers for the numpy array (variables start with *raw_*).

#### trainer.py

To use GPUs

1. Uncomment line 83: `self._net = torch.nn.DataParallel(self._net, device_ids=[0, 1, 2, 3])` and specify amount of GPUs to use.
2. Change line 175 and 176 to contain `module`:         
   > `175` &ensp; self._net.module.save(self._output_dir, TrainingConstants.NET_SAVE_FNAME, str(epoch) + '_') <br>
   > `176` &ensp; &ensp; self._net.module.save(self._output_dir, TrainingConstants.NET_SAVE_FNAME, 'final_')
3. Specify amount of workers for loading data in line 70: `num_workers=1`. On the DGX limit is 6 to not disturb others.  

#### dataset.py

Similar to `moments.py`: update code if data data is spread across several directories.

#### train.py

Command-line options for `train.py` are:
* `'data_dir', type=str` -- Path to the training data.
* `'--num_epochs', type=int, default=TrainingConstants.NUM_EPOCHS` -- Number of training epochs.
* `'--total_size', type=float, default=TrainingConstants.TOTAL_SIZE` -- The proportion of the data to use.
* `'--val_size', type=float, default=TrainingConstants.VAL_SIZE` -- The proportion of the data to use for validation.
* `'--bsz', type=int, default=TrainingConstants.BSZ` -- Training batch size.
* `'--base_lr', type=float, default=TrainingConstants.BASE_LR` -- Base learning rate.
* `'--lr_step_size' type=int default=TrainingConstants.LR_STEP_SIZE` -- Step size for learning rate in epochs.
* `'--lr_decay_rate', type=float, default=TrainingConstants.LR_DECAY_RATE` -- Decay rate for learning rate at every --lr_step_size.
* `'--log_interval', type=int, default=TrainingConstants.LOG_INTERVAL` -- Log interval in batches. 
* `'--cuda', action='store_true'` -- Enable CUDA support and utilize GPU devices.
* `'--output_dir',  type=str, default=TrainingConstants.OUTPUT_DIR` -- Directory to output logs and trained model to.
* `'--net_name', type=str, default=TrainingConstants.NET_NAME` -- Name of network.

#### eval_policy.py

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

#### get_metrics.py

Get the final metrics, i.e. the average total coverage, average diversity and average total water use for the garden runs.
