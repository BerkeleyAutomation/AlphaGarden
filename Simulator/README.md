# simulator

As the name indicates, the simulator package contains the code for simulating polyculture farming. This README 
supports the in-line documentation and serves as a high-level documentation of the class files.

## Files structure

The most important (class) files are mentioned in this table.

| **Files and classes**                                                                        | **Description**               |
| ------------------------------------------------------------------------------------- |-----------------------------|
| [SimAlphaGardenWrapper.py](AlphaGarden/Simulator/simulator/SimAlphaGardenWrapper.py)  |AlphaGarden's wrapper for Gym, inheriting basic functions from the WrapperEnv.|
| [garden.py](AlphaGarden/Simulator/simulator/garden.py)                                |Model for garden. |
| [plant_stage.py](AlphaGarden/Simulator/simulator/plant_stage.py)                      |Modeling plant stages in bio standard life cycle trajectory.|
| [plant_type.py](AlphaGarden/Simulator/simulator/plant_type.py)                        |High-level structure for plant types available in garden.|
| [plant.py](AlphaGarden/Simulator/simulator/plant.py)                                  |Model for plants. |
| [baseline_policy.py](AlphaGarden/Simulator/simulator/baselines/baseline_policy.py)    |                         |