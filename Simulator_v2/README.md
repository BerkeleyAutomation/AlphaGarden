# AlphaGarden Simulator v2

## Quickstart

```
$    cd simulatorv2
$    python3 run_simulation.py  
```

The simulation can be run with a number of flags:

```
--mode [a|s|p] 

[a]: show full animation
[s]: save data to produce animations 
[p]: show plots of plant behaviors


--setup [random|csv|]

[random]: place plants randomly across the garden 
[csv]: read in plant locations from the .csv file given by the --csv_path arg 
[{preset}] see a full list of presets in simulator_presets.py

```
