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
to learn a control policy for the garden that is reliable and sustainable. In this section the instructions are first
given for the simulator and afterwards for the adaptive automation policy presented in the paper.

### Simulation Experiments

The 

List of used plant types and their important parameters. AlphaGarden/Simulator/simulator/plant_presets.py

| Plant Type      | Mean Germination Time (days)  | Mean Maturation Time (days)  | Mean Mature Radius (inches) | k<sub>1</sub> | k<sub>2</sub> |
| --------------- |:-----------------------------:|:----------------------------:|:---------------------------:|:-------------:| -------------:|
| Generic invasive| | | | |
| Bok Choy        |7.5                            |45                            |3                            |0.33           |0.389          |
| Basil           |7.5                            |62.5                          |4.5                          |0.33           |0.389          |
| Lavender        |17.5                           |145                           |10.5                         |0.428          |0.455          |
| Parsley         |24.5                           |80                            |5.25                         |0.33           |0.389          |
| Sage            |15.5                           |730                           |15                           |0.428          |0.455          |
| Chives          |18                             |90                            |3.75                         |0.33           |0.389          |
| Cilantro        |8.5                            |67.5                          |2                            |0.33           |0.389          |
| Dill            |8.5                            |70                            |6.75                         |0.33           |0.389          |
| Fennel          |10                             |65                            |5.5                          |0.33           |0.389          |
| Nasturtium      |11                             |60                            |5.5                          |0.428          |0.455          |
| Marigold        |7.5                            |50                            |3.5                          |0.267          |0.38           |
| Calendula       |8.5                            |50                            |6                            |0.267          |0.38           |
| Borage          |10                             |56                            |10                           |0.267          |0.38           |
| ...     | ... |... |... |... | ... |

### Adaptive Automation Policy


## Troubleshooting

## Notes and Miscellaneous

## Building the Extension Bundles

## Next Steps

## Credits

## Feedback

## License

### Code

### Images
