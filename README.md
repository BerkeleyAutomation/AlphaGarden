# Overview 
In this repository, you will find code for AlphaGardenSim and for AlphaGarden. AlphaGarden is an autonomous polyculture garden located in Berkeley, CA that uses AlphaGardenSim to decide watering and pruning actions. Papers explaining the functionality of AlphaGarden can be found [here](https://ieeexplore.ieee.org/document/9216984) and [here](https://ieeexplore.ieee.org/document/9684020). 

The [Learning](https://github.com/BerkeleyAutomation/AlphaGarden/Learning) and [Simulator](https://github.com/BerkeleyAutomation/AlphaGarden/Simulator) folders contain functionality for AlphaGardenSim. Please see the rest of this README for information on how to use AlphaGardenSim.

The [State-Estimation](https://github.com/BerkeleyAutomation/AlphaGarden/State-Estimation) folder contains functionally for the Plant Phenotyping NN that identifies a plant's type, Bounding Disk Tracking algorithm that is used to identify a plant's radius, and a Prune Point Identification NN that identifies the centers of leaves. The [Actuation](https://github.com/BerkeleyAutomation/AlphaGarden/Actuation) folder includes functionality for how we interface with the [FarmBot](https://farm.bot/), a commercial gantry system for gardens. README's for these methods can be found in their respective folders. 

# AlphaGardenSim

AlphaGardenSim is a fast, first order open-access simulator that integrates single plant growth models with inter-plant 
competition for sun light and water. The aim of the simulator is to support learning control policies for cultivating
a polyculture garden. More on this [here](http://alphagarden.org/). 
The simulator implements a custom [OpenAI gym](https://gym.openai.com/) reinforcement learning environment for this 
polyculture farming setup.

![AlphaGarden Watercolering](store-assets/watercolorAlphaGarden.png)

### Table of Contents
**[Installation Instructions](#installation-instructions)**<br>
**[Usage Instructions](#usage-instructions)**<br>
**[License](#next-steps)**<br>

## Installation Instructions and Quickstart

For now the *AlphaGardenSim* module is distributed with two parts: [Learning](https://github.com/BerkeleyAutomation/AlphaGarden/Learning) 
containing the *simalphagarden* and *wrapperenv* packages and [Simulator](https://github.com/BerkeleyAutomation/AlphaGarden/Simulator) 
containing the *simulator* package.

Install the required pip packages and the mentioned packages from above:

1. `git clone` the repository
2. Open the `AlphaGarden` [repository](https://github.com/BerkeleyAutomation/AlphaGarden/)
3. Run ```pip install -r requirements.txt ```. Make sure you are using pip 20.0.2. Version 20.1 currently is not supported.

To run the simulator and collect data:

4. Move to the Learning folder with `cd Learning/`
5. Run `python data_collection.py`

See the **[Usage Instructions](#usage-instructions)** for more details on how to use the simulator.

### Tests

Currently no tests are included. Tests may be added in the future.

### Built With

* [gym](https://gym.openai.com/) - Toolkit for developing and comparing reinforcement learning algorithms

## Usage Instructions

The simulator models the inter-plant dynamics and competition for light and water. A main idea is to use the simulator
to learn a control policy for the garden that is reliable and sustainable.

### Simulation Experiments

To run your own experiments or reproduce the experiments from the paper follow these instructions.

* Some of the parameters described in the experimental setup are stored in the [sim_global.py](AlphaGarden/Simulator/simulator/sim_globals.py) file 

* Experimental data can be generated with the [data_collection.py](AlphaGarden/Learning/data_collection.py) module. Further important parameters are defined in this file.

* List of 13 edible plant types used with different germination times, maturation times and growth rates, sampled from plant-specific Gaussian distributions.
Plants are modeled with the [Plant](AlphaGarden/Simulator/simulator/plant.py) class and the data can be found [here](AlphaGarden/Simulator/simulator/plant_presets.py)

| Plant Type       | Mean Germination Time (days)  | Mean Maturation Time (days)  | Mean Growth Potential |  c<sub>1</sub> (Water Use Efficiency/Growth Rate) |
|:----------------:|:-----------------------------:|:----------------------------:|:---------------------------:|:-------------:|
| Kale         |              7             |              55              |          50                 |      0.28     |
| Borage         |              7             |           55                 |          50                 |     0.24      |
| Turnip         |                 7          |               47             |          50                 |     0.28      |
| Swiss Chard         |          7                 |          50                  |            47               |     0.26      |
| Radicchio         |                9           |                55            |            43               |      0.30     |
| Arugula         |               8            |               52             |             40              |      0.40     |
| Red Lettuce         |              12             |              50              |                28           |      0.30     |
| Green Lettuce         |            9               |           52                 |           27                |     0.25      |
| Cilantro         |           10                |                65            |           20                |     0.32      |
| Sorrel         |            15               |              70              |            8               |     0.32      |
| ...              | ...                           |...                           |...                          |...            |

### Evaluation

To evaluate automation policies with the simulator:

1. Move to the Learning folder with `cd Learning/`
2. Run `python eval_policy.py` to evaluate the the baseline policy with serial execution.

Other command-line options for `eval_policy.py` are:
* `'-t', '--tests', type=int, default=1` -- Number of evaluation trials.
* `'-n', '--net', type=str, default='/'` -- To evaluate a learned policy supply the trained params.
* `'-m', '--moments', type=str, default='/'` -- To evaluate a learned policy supply the moments of the dataset used to train the policy.
* `'-s', '--seed', type=int, default=0` -- Numpy's random seed.
* `'-p', '--policy', type=str, default='b'` -- baseline [b], naive baseline [n], learned [l], irrigation [i].
* `'--multi', action='store_true'` -- Enable multiprocessing.
* `'-l', '--threshold', type=float, default=1.0` -- Prune threshold
* `'-d', '--days', type=int, default=100` -- Garden days

## License

### Code

MIT License

Copyright (c) 2020 UC Berkeley AUTOLAB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Images
Watercolor by Chelsea Qiu and Sarah Newman, Copyright (c) 2019
