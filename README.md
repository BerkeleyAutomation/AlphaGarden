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

| Plant Type       | Mean Germination Time (days)  | Mean Maturation Time (days)  | Mean Mature Radius (inches) | k<sub>1</sub> | k<sub>2</sub> |
|:----------------:|:-----------------------------:|:----------------------------:|:---------------------------:|:-------------:|:-------------:|
| Bok Choy         |7.5                            |45                            |3                            |0.33           |0.389          |
| Basil            |7.5                            |62.5                          |4.5                          |0.33           |0.389          |
| Lavender         |17.5                           |145                           |10.5                         |0.428          |0.455          |
| Parsley          |24.5                           |80                            |5.25                         |0.33           |0.389          |
| Sage             |15.5                           |730                           |15                           |0.428          |0.455          |
| Chives           |18                             |90                            |3.75                         |0.33           |0.389          |
| Cilantro         |8.5                            |67.5                          |2                            |0.33           |0.389          |
| Dill             |8.5                            |70                            |6.75                         |0.33           |0.389          |
| Fennel           |10                             |65                            |5.5                          |0.33           |0.389          |
| Nasturtium       |11                             |60                            |5.5                          |0.428          |0.455          |
| Marigold         |7.5                            |50                            |3.5                          |0.267          |0.38           |
| Calendula        |8.5                            |50                            |6                            |0.267          |0.38           |
| Borage           |10                             |56                            |10                           |0.267          |0.38           |
| Generic invasive |                               |                              |                             |               |               |
| ...              | ...                           |...                           |...                          |...            | ...           |

### Evaluation

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
