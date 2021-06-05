from simulator.sim_globals import ROWS, COLS, STEP, SECTOR_ROWS, SECTOR_COLS, PRUNE_WINDOW_ROWS, PRUNE_WINDOW_COLS, PRUNE_RATE, IRR_THRESHOLD
from simulator.plant_presets import PLANT_TYPES
from simulator.plant_type import PlantType
from simulator.garden_state import GardenState
from simulator.garden import Garden
import numpy as np
import pickle

''' From garden.py '''
def compute_growth_map():
    growth_map = []
    for i in range(max(COLS, ROWS) // 2 + 1):
        for j in range(i + 1):
            points = set()
            points.update(((i, j), (i, -j), (-i, j), (-i, -j), (j, i), (j, -i), (-j, i), (-j, -i)))
            growth_map.append((STEP ** 0.5 * np.linalg.norm((i, j)), points))
    growth_map.sort(key=lambda x: x[0])
    return growth_map

def add_plant(plant, id, plants, plant_types, plant_locations, grid, plant_grid, leaf_grid):
    """ Add plants to garden's grid locations.
    Args:
        plant: Plants objects for Garden.
    """
    if (plant.row, plant.col) in plant_locations:
        print(
            f"[Warning] A plant already exists in position ({plant.row, plant.col}). The new one was not planted.")
    else:
        plant.id = id
        plants[plant_types.index(plant.type)][plant.id] = plant
        plant_locations[plant.row, plant.col] = True
        grid[plant.row, plant.col]['nearby'].add((plant_types.index(plant.type), plant.id))
        plant_grid[plant.row, plant.col, plant_types.index(plant.type)] = 1
        leaf_grid[plant.row, plant.col, plant_types.index(plant.type)] += 1

def enumerate_grid(grid):
    for i in range(0, len(grid)):
        for j in range(len(grid[i])):
            yield (grid[i, j], (i, j))
                    
def compute_plant_health(grid, grid_shape, plants):
    """ Compute health of the plants at each grid point.
    Args:
        grid_shape (tuple of (int,int)): Shape of garden grid.
    Return:
        Grid shaped array (M,N) with health state of plants.
    """
    plant_health_grid = np.empty(grid_shape)
    for point in enumerate_grid(grid):
        coord = point[1]
        if point[0]['nearby']:

            tallest_height = -1
            tallest_plant_stage = 0
            tallest_plant_stage_idx = -1

            for tup in point[0]['nearby']:
                plant = plants[tup[0]][tup[1]]
                if plant.height > tallest_height:
                    tallest_height = plant.height
                    tallest_plant_stage = plant.stages[plant.stage_index]
                    tallest_plant_stage_idx = plant.stage_index

            if tallest_plant_stage_idx in [-1, 3, 4]:
                plant_health_grid[coord] = 0
            elif tallest_plant_stage_idx == 0:
                plant_health_grid[coord] = 2
            elif tallest_plant_stage_idx in [1, 2]:
                if tallest_plant_stage.overwatered:
                    plant_health_grid[coord] = 3
                elif tallest_plant_stage.underwatered:
                    plant_health_grid[coord] = 1
                else:
                    plant_health_grid[coord] = 2

    return plant_health_grid

def copy_garden(garden_state, rows, cols, sector_row, sector_col, prune_win_rows, prune_win_cols, step, prune_rate):
    garden = Garden(
               garden_state=garden_state,
                N=rows,
                M=cols,
                sector_rows=sector_row,
                sector_cols=sector_col,
                prune_window_rows=prune_win_rows,
                prune_window_cols=prune_win_cols,
                irr_threshold=IRR_THRESHOLD,
                step=step,
                prune_rate = prune_rate,
                animate=False)
    return garden

# INPUT {type: {(x, y), radius}, ..., {}}
# real_data = {
#     'borage': {
#         ((2, 2), 10),
#         ((5, 5), 10),
#     },
#     'arugula': {
#         ((60, 60), 10),
#         ((100, 100), 10),
#     },
# }

# real_data = {'kale': {((42, 70), 23), ((112, 102), 25)}, 'swiss_chard': {((101, 80), 3), ((111, 21), 3)}, 'red_lettuce': {((85, 105), 0), ((55, 98), 0)}, 'cilantro': {((62, 17), 11), ((136, 19), 9)}, 'radicchio': {((85, 26), 18), ((36, 29), 11)}, 'arugula': {((103, 39), 14), ((128, 43), 6)}, 'sorrel': {((24, 81), 0), ((24, 56), 1)}, 'borage': {((74, 78), 18), ((87, 52), 24)}, 'turnip': {((119, 62), 24), ((59, 37), 25)}, 'green_lettuce': {((30, 113), 24), ((125, 25), 24)}}
real_data = {'red_lettuce': {((112, 19), 9), ((96, 83), 9)}, 'kale': {((40, 71), 25), ((112, 102), 25)}, 'radicchio': {((81, 21), 14), ((32, 29), 3)}, 'cilantro': {((136, 16), 8), ((66, 16), 8)}, 'swiss_chard': {((81, 105), 25), ((50, 101), 25)}, 'green_lettuce': {((29, 114), 12), ((122, 24), 12)}, 'borage': {((72, 78), 24), ((86, 55), 25)}, 'turnip': {((115, 62), 25), ((57, 38), 25)}, 'arugula': {((133, 41), 10), ((103, 34), 15)}, 'sorrel': {((24, 81), 0), ((24, 56), 3)}}

timestep = 30

plant_type = PlantType()
plant_types = plant_type.plant_names
plant_objs = plant_type.get_plant_seeds(0, ROWS, COLS, SECTOR_ROWS, SECTOR_COLS,
                                        start_from_germination=False, existing_data=real_data,
                                        timestep=timestep)

plants = [{} for _ in range(len(plant_types))]

grid = np.empty((ROWS, COLS), dtype=[('water', 'f'), ('health', 'i'), ('nearby', 'O')])
grid['water'] = np.random.normal(0.4, 0.1, grid['water'].shape)
for i in range(ROWS):
    for j in range(COLS):
        grid[i, j]['nearby'] = set()

plant_grid = np.zeros((ROWS, COLS, len(plant_types)))

plant_prob = np.zeros((ROWS, COLS, 1 + len(plant_types)))

leaf_grid = np.zeros((ROWS, COLS, len(plant_types)))

plant_locations = {}

id_ctr = 0
for plant in plant_objs:
    add_plant(plant, id_ctr, plants, plant_types, plant_locations, grid, plant_grid, leaf_grid)
    id_ctr += 1
    
grid['health'] = compute_plant_health(grid, grid['health'].shape, plants)

growth_map = compute_growth_map()

radius_grid = np.zeros((ROWS, COLS, 1))
for p_type in real_data:
    for plant in real_data[p_type]:
        r, c = plant[0]
        radius = plant[1]
        radius_grid[r, c, 0] = radius 

garden_state = GardenState(plants, grid, plant_grid, plant_prob, leaf_grid, plant_type,
                           plant_locations, growth_map, radius_grid, timestep, existing_data=True)
garden_copy = copy_garden(garden_state=garden_state, rows=ROWS, cols=COLS, sector_row=SECTOR_ROWS,
                          sector_col=SECTOR_COLS, prune_win_rows=PRUNE_WINDOW_ROWS,
                          prune_win_cols=PRUNE_WINDOW_COLS, step=STEP, prune_rate=PRUNE_RATE)
pickle.dump([garden_copy, plant_type], open("garden_copy.pkl", "wb")) 
print("SAVED!")