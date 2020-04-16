# AlphaGardenSim

AlphaGardenSim is a fast, first order open-access simulator that integrates single plant growth models with inter-plant 
competition for sun light and water. The aim of the simulator is to support learning a control policy for cultivating
a polyculture garden. More on this [here](https://goldberg.berkeley.edu/art/AlphaGarden/). 
The simulator implements a custom [OpenAI gym](https://gym.openai.com/) reinforcement learning environment for this 
polyculture farming setup. More on the simulator can be found in the [paper]().

![AlphaGarden Watercolering](store-assets/watercolorAlphaGarden.png)

### Table of Contents
**[Installation Instructions](#installation-instructions)**<br>
**[Usage Instructions](#usage-instructions)**<br>
**[Troubleshooting](#troubleshooting)**<br>
**[Notes and Miscellaneous](#notes-and-miscellaneous)**<br>
**[Building the Extension Bundles](#building-the-extension-bundles)**<br>
**[Next Steps, Credits, Feedback, License](#next-steps)**<br>

## Installation Instructions

For now the *AlphaGardenSim* module is distributed with two parts: [Learning](https://github.com/BerkeleyAutomation/AlphaGarden/Learning) 
containing the *simalphagarden* and *wrapperenv* packages and [Simulator](https://github.com/BerkeleyAutomation/AlphaGarden/Simulator) 
containing the *simulator* package.

Install the required pip packages and the mentioned packages from above:

Run ```pip install -r requirements.txt ``` inside [AlphaGarden](https://github.com/BerkeleyAutomation/AlphaGarden/).

See the **[Usage Instructions](#usage-instructions)** for more details on how to use the simulator.

### Tests

Currently no tests are included. Tests may be added in the future.

### Built With

* [gym](https://gym.openai.com/) - Toolkit for developing and comparing reinforcement learning algorithms
* [TODO add other]

## Usage Instructions

The simulator models the inter-plant dynamics and competition for light and water. A main idea is to use the simulator
to learn a control policy for the garden that is reliable and sustainable. In this section the usage instructions for are
described first for the simulator and afterwards for the adaptive automation policy presented in the paper.

### Simulation Experiments

List of important variables for the experiments and their location in the simulator.

| Variable  | Description  | File location |
| --------- |:-----------------------------:|:-----------------------------:  |
| D[k] 		|Vector of K-2 plant types (+ earth, manual invasive and unknown = k) |
|p(x,y)  	|The garden is defined on a regular grid of 150 x 300 points|
|s(x,y,t)   |Due to limits on robot precision, we divide Garden into sectors of size 15 x 30
|d(x,y,t)   |At each point at time at time t (measured in days), we estimate the local condition with a distribution over D |
|s(x,y,t)   |For all seedlings, we record their initial location and time when they sprout |
| A[.]		|Vector of action types: A[0] = no action, A[1] = irrigate with one fixed burst, A[2] = prune |
|a(x,y,t)   |Action performed at time t at the center of sector s(i,j)  = 0,1, or 2 |
|w(x,y,t)   |Estimated moisture at each point based on past watering actions and our model of radial distribution, drainage, plant update, and evaporation.|
|h(x,y,t)   |Health of plants at point (x,y) at time t, based on history of d(x,y) and w(x,y) and plant parameters for d(x,y,t): real value in range [0-1]  (0 is dead).|
|P[k,t]		|Vector of total population of each plant type k, by summing over d(x,y,t)|
|D[t]		|Diversity of overall garden at time t, maximal when P[i,t] = P[j,t] for all i,j > 1.|
|C[t]		|Coverage at time t is the sum over all x,y where d(x,y)>0 (not earth)|
|S[t]		|Sustainability at time t in terms of irrigation savings over uniform baseline|
|O(i,j,t)	|Observation at sector s(i,j,t)|

List of used plant types and their important parameters.

| Plant Type      | Mean Germination Time (days)  | Mean Maturation Time (days)  | Mean Mature Radius (inches) | k<sub>1</sub> | k<sub>2</sub> |
| --------------- |:-----------------------------:| :---------------------------:|  :-------------------------:|  :-----------:| -------------:|
| manual invasive | | | | |
| | | | | |
| | | | | |

### Adaptive Automation Policy



![Table of Plants ](https://raw.github.com/BerkeleyAutomation/AlphaGarden/store-assets/plantTable.png)

## Troubleshooting

## Notes and Miscellaneous

## Building the Extension Bundles

## Next Steps

## Credits

## Feedback

## License

### Code

### Images
