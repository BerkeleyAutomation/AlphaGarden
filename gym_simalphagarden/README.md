Contents
--------

The RL training environment setup consists of:

1.  A custom OpenAI gym environment.

2.  A user defined configuration file for the environment.

3.  A python wrapper for a plant simulator that inherits the wrapperenv interface.

Running
-------

Steps to use the gym environment:

1.  Run ```pip install -e .``` inside [gym\_simalphagarden/](https://github.com/w07wong/SimAlphaGarden/tree/master/gym_simalphagarden).

2.  Create a wrapper environment for your simulator that implements a [take\_action()](https://github.com/w07wong/SimAlphaGarden/blob/master/gym_simalphagarden/wrapperenv/wrapper_interface.py#L17) method and [reset()](https://github.com/w07wong/SimAlphaGarden/blob/master/gym_simalphagarden/wrapperenv/wrapper_interface.py#L21) method. Have the class ```import wrapperenv``` and inherit from ```wrapperenv.WrapperEnv```.

1.  The interface can be found [here](https://github.com/w07wong/SimAlphaGarden/blob/master/gym_simalphagarden/wrapperenv/wrapper_interface.py).

4.  Create a configuration file to customize your environment. Instructions below.

5.  In the file where you want to train a model, 'import simalphagarden' and pass in the wrapper environment and config file like [so](https://github.com/w07wong/SimAlphaGarden/blob/master/AquaCropOS_v50a/gym_test.py#L11).

Configuration File
------------------

The configuration file for the simalphagarden environment currently specifies the reward range, range of discrete actions and the observation space.

Create your config file with the following structure. An example can be found [here](https://github.com/w07wong/SimAlphaGarden/blob/master/AquaCropOS_v50a/config/aquacropos_config.ini).

```
[reward]
# the lowest/highest value a reward can take
low = <float> 
high = <float> 

[action]
# actions are discrete numbers from [0, n) with n specified below
range = <int> 

[obs]
# the lowest/highest value an observation can have
low = <float>
high = <float>
# currently, only supports 2 observations. TODO: modify the environment to support more
shape_x = 2
shape_y = 2
```

Example
-------

A setup for AquaCropOS is in [this](https://github.com/w07wong/SimAlphaGarden/tree/master/AquaCropOS_v50a) directory.

The simulator's wrapper is [here](https://github.com/w07wong/SimAlphaGarden/blob/master/AquaCropOS_v50a/aquacropos_wrapper.py#L8) and implements the [WrapperEnv](https://github.com/w07wong/SimAlphaGarden/blob/master/gym_simalphagarden/wrapperenv/wrapper_interface.py) interface.

We run PPO with the simalphagarden gym [here](https://github.com/w07wong/SimAlphaGarden/blob/master/AquaCropOS_v50a/gym_test.py).
