import numpy as np
import yaml
import os

LIGHT_USE = 1.0

PLANT_TYPES = ['green_lettuce', 'red_lettuce', 'borage', 'swiss_chard', 'kale',
               'cilantro', 'radicchio', 'turnip', 'invasive']

GARDEN_SIZE = 100
BORDER = 0
AMOUNT_PLANTS = 3
AMOUNT_PLANTS_TYPES = 3

"""coordinate_transfer = {
    'borage': [(65, 120), (121, 73)],
    'cilantro': [(137, 36), (14, 31)],
    'radicchio': [(90, 24), (24, 84)],
    'kale': [(56, 35), (94, 97)],
    'green_lettuce': [(24, 16), (116, 18)],
    'red_lettuce': [(90, 135), (134, 22)],
    'swiss_chard': [(27, 55), (121, 121)],
    #'turnip': [(84, 58), (34, 116)],
    'invasive': [(84, 58), (34, 116)]
}"""

"""coordinate_transfer = {'green_lettuce': [(30, 30)],
                       'red_lettuce': [(86, 30)],
                       'cilantro': [(30, 85)],
                       #'radicchio': [(83, 81)],
                       'invasive': [(83, 81)],
                       }"""

"""coordinate_transfer = {
    'kale': [(56, 35)],
    'green_lettuce': [(24, 16)],
    'red_lettuce': [(90, 135)],
    'swiss_chard': [(27, 55)],
    'turnip': [(84, 58)],
    #'invasive': [(56, 35)]
}"""
"""coordinate_transfer = {'A': [(14, 14)],
                       'B': [(42, 14)],
                       'C': [(14, 42)],
                       'D': [(42, 42)],
                       'E': [(28, 28)]}"""

coordinate_transfer = {'green_lettuce': [(28, 28)],
                       'red_lettuce': [(72, 28)],
                       #'invasive': [(50, 66)],
                        'turnip': [(50, 66)]
                       }

"""coordinate_transfer = {'A': [(10, 7)],
                       'B': [(18, 7)],
                       }"""

OUTER_RADIUS = {
    'borage': (34, 2),
    'cilantro': (20, 2),
    'radicchio': (22, 2),
    'kale': (42, 2),
    'green_lettuce': (25, 2),
    'red_lettuce': (20, 2),
    'swiss_chard': (33, 2),
    'turnip': (35, 2),
    'invasive': (55, 2),
    'A': (14, 2),
    'B': (14, 2),
    'C': (14, 2),
    'D': (14, 2),
    'E': (14, 2),

    'lower_steps': 4,
    'upper_steps': 4
}

WATER_USE = {
    'borage': 0.165,
    'cilantro': 0.13,
    'radicchio': 0.145,
    'kale': 0.165,
    'green_lettuce': 0.14,
    'red_lettuce': 0.145,
    'swiss_chard': 0.1699,
    'turnip': 0.195,
    'invasive': 0.22,
    'A': 0.15,
    'B': 0.25,
    'C': 0.15,
    'D': 0.15,
    'E': 0.15,
}

MATUR_TIME = {
    'borage': (60, 2),
    'cilantro': (70, 2),
    'radicchio': (55, 2),
    'kale': (70, 2),
    'green_lettuce': (60, 2),
    'red_lettuce': (70, 2),
    'swiss_chard': (60, 2),
    'turnip': (70, 2),
    'invasive': (55, 2),
    'A': (50, 5),
    'B': (50, 5),
    'C': (50, 5),
    'D': (50, 5),
    'E': (50, 5),

    'range_min': 1,
    'upper_steps': 3
}

GERM_TIME = {
    'borage': (4, 1),
    'cilantro': (8, 1),
    'radicchio': (7, 1),
    'kale': (6, 1),
    'green_lettuce': (8, 1),
    'red_lettuce': (10, 1),
    'swiss_chard': (4, 1),
    'turnip': (5, 1),
    'invasive': (2, 1),
    'A': (5, 1),
    'B': (5, 1),
    'C': (5, 1),
    'D': (5, 1),
    'E': (5, 1),

    'range_min': 1,
    'upper_steps': 5
}

WAIT_TIME = {
    'borage': (5, 2),
    'cilantro': (3, 2),
    'radicchio': (10, 2),
    'kale': (3, 2),
    'green_lettuce': (5, 2),
    'red_lettuce': (3, 2),
    'swiss_chard': (8, 2),
    'turnip': (3, 2),
    'invasive': (3, 2),
    'A': (10, 2),
    'B': (10, 2),
    'C': (10, 2),
    'D': (10, 2),
    'E': (10, 2),

    'range_min': 1.0,
    'range_max': 100.0
}

WILT_TIME = {
    'borage': (18, 2),
    'cilantro': (16, 2),
    'radicchio': (18, 2),
    'kale': (3, 2),
    'green_lettuce': (18, 2),
    'red_lettuce': (16, 2),
    'swiss_chard': (18, 2),
    'turnip': (16, 2),
    'invasive': (10, 2),
    'A': (20, 2),
    'B': (20, 2),
    'C': (20, 2),
    'D': (20, 2),
    'E': (20, 2),

    'range_min': 1.0,
    'upper_steps': 2
}



default_config = {
    'amount_plants': {'default_value': AMOUNT_PLANTS},
    'amount_plant_types': {'default_value': AMOUNT_PLANTS_TYPES},
    'garden_length': {'default_value': GARDEN_SIZE},
    'garden_width': {'default_value': GARDEN_SIZE},
    'sector_rows': {'default_value': 15},
    'sector_cols': {'default_value': 30},
    'garden_days': {'default_value': 100},

    'max_water_content': {'default_value': 0.3},
    'max_nutrient_content': {'default_value': 1.0},
    'permanent_wilting_point': {'default_value': 0.0},
    'init_water_mean': {'default_value': 0.2},
    'init_water_scale': {'default_value': 0.04},
    'evaporation_percent_mean': {'default_value': 0.04},
    'evaporation_percent_scale': {'default_value': 0.0054},
    'irrigation_amount': {'default_value': 0.0002},
    'water_threshold': {'default_value': 1.0},

    'irr_health_window_width': {'default_value': 9},
    'prune_window_rows': {'default_value': 5},
    'prune_window_cols': {'default_value': 5},
    'prune_rate': {'default_value': 0.15},
    'prune_delay': {'default_value': 20},
    'reference_outer_radii': {
        'default_value': [],
        'dtype': 'np.float32',
        'randomize': {'distribution_type': 'truncated_normal',
                      'scale': [],
                      'range_min': [],
                      'range_max': []
                      }
    },
    'common_names': {'default_value': []},
    'x_coordinates': {
        'default_value': [],
        'dtype': 'np.float32',
        'randomize': {'distribution_type': 'random_coordinate', 'range_min': 0 + BORDER,
                      'range_max': GARDEN_SIZE - BORDER, 'shape': AMOUNT_PLANTS}},
    'y_coordinates': {
        'default_value': [],
        'dtype': 'np.float32',
        'randomize': {'distribution_type': 'random_coordinate', 'range_min': 0 + BORDER,
                      'range_max': GARDEN_SIZE - BORDER, 'shape': AMOUNT_PLANTS}},
    'current_outer_radii': {
        'default_value': [],
        'dtype': 'np.float32'
    },
    'germination_times': {
        'default_value': [],
        'dtype': 'np.int',
        'randomize':
            {'distribution_type': 'truncated_normal',
             'scale': [],
             'range_min': [],
             'range_max': []
             }
    },
    'maturation_times': {
        'default_value': [],
        'dtype': 'np.int',
        'randomize': {
            'distribution_type': 'truncated_normal',
            'scale': [],
            'range_min': [],
            'range_max': []
        }
    },
    'waiting_times': {
        'default_value': [],
        'dtype': 'np.int',
        'randomize': {
            'distribution_type': 'truncated_normal',
            'scale': [],
            'range_min': [],
            'range_max': []
        }
    },
    'wilting_times': {
        'default_value': [],
        'dtype': 'np.int',
        'randomize': {
            'distribution_type': 'truncated_normal',
            'scale': [],
            'range_min': [],
            'range_max': []
        }
    },
    'light_use_efficiencies': {
        'default_value': [],
        'dtype': 'np.float32'
    },
    'water_use_efficiencies': {
        'default_value': [],
        'dtype': 'np.float32'
    },
    'nutrients_use_efficiencies': {
        'default_value': [],
        'dtype': 'np.float32'
    },
    'tau': {'default_value': 0},
    'overwatered_time_threshold': {'default_value': 5},
    'underwatered_time_threshold': {'default_value': 5},
    'overwatered_threshold': {'default_value': 100},
    'underwaterd_threshold': {'default_value': 0.01}
}


def create_config_yaml():
    common = []
    x_c = []
    y_c = []
    out_rad_mean = []
    out_rad_scale = []
    out_rad_min = []
    out_rad_max = []
    cur_rad = []
    matur_time_m = []
    matur_time_s = []
    matur_time_min = []
    matur_time_max = []
    germ_time_s = []
    germ_time_m = []
    germ_time_min = []
    germ_time_max = []

    wait_time_m = []
    wait_time_s = []
    wait_time_min = []
    wait_time_max = []

    wilt_time_m = []
    wilt_time_s = []
    wilt_time_min = []
    wilt_time_max = []

    c_water = []
    c_light = []
    is_invasive = False
    randomize_coordinates = False
    for key, val in coordinate_transfer.items():
        for (x, y) in val:
            if key == 'invasive':
                is_invasive = True
            o_r = OUTER_RADIUS[key][0]
            out_rad_min.append(o_r - OUTER_RADIUS['lower_steps'])
            out_rad_max.append(o_r + OUTER_RADIUS['upper_steps'])
            out_rad_mean.append(o_r)
            out_rad_scale.append(OUTER_RADIUS[key][1])

            cur_rad.append(1)

            m_t = MATUR_TIME[key][0]
            matur_time_m.append(m_t)
            matur_time_s.append(MATUR_TIME[key][1])
            matur_time_min.append(MATUR_TIME['range_min'])
            matur_time_max.append(m_t + MATUR_TIME['upper_steps'])

            g_t = GERM_TIME[key][0]
            germ_time_m.append(g_t)
            germ_time_s.append(GERM_TIME[key][1])
            germ_time_min.append(GERM_TIME['range_min'])
            germ_time_max.append(g_t + GERM_TIME['upper_steps'])

            wait = WAIT_TIME[key][0]
            wait_time_m.append(wait)
            wait_time_s.append(WAIT_TIME[key][1])
            wait_time_min.append(WAIT_TIME['range_min'])
            wait_time_max.append(WAIT_TIME['range_max'])

            wilt = WILT_TIME[key][0]
            wilt_time_m.append(wilt)
            wilt_time_s.append(WILT_TIME[key][1])
            wilt_time_min.append(WILT_TIME['range_min'])
            wilt_time_max.append(wilt + WILT_TIME['upper_steps'])

            c_light.append(1.0)
            c_water.append(WATER_USE[key])
            common.append(key)

            x_c.append(x)
            y_c.append(y)

    # default_config['amount_plants'] = num_plants
    default_config['reference_outer_radii']['default_value'] = out_rad_mean
    default_config['reference_outer_radii']['randomize']['scale'] = out_rad_scale
    default_config['reference_outer_radii']['randomize']['range_min'] = out_rad_min
    default_config['reference_outer_radii']['randomize']['range_max'] = out_rad_max
    # default_config['amount_plant_types'] = len(PLANT_TYPES)
    default_config['common_names']['default_value'] = common

    default_config['germination_times']['default_value'] = germ_time_m
    default_config['germination_times']['randomize']['scale'] = germ_time_s
    default_config['germination_times']['randomize']['range_min'] = germ_time_min
    default_config['germination_times']['randomize']['range_max'] = germ_time_max

    default_config['maturation_times']['default_value'] = matur_time_m
    default_config['maturation_times']['randomize']['scale'] = matur_time_s
    default_config['maturation_times']['randomize']['range_min'] = matur_time_min
    default_config['maturation_times']['randomize']['range_max'] = matur_time_max

    default_config['waiting_times']['default_value'] = wait_time_m
    default_config['waiting_times']['randomize']['scale'] = wait_time_s
    default_config['waiting_times']['randomize']['range_min'] = wait_time_min
    default_config['waiting_times']['randomize']['range_max'] = wait_time_max

    default_config['wilting_times']['default_value'] = wilt_time_m
    default_config['wilting_times']['randomize']['scale'] = wilt_time_s
    default_config['wilting_times']['randomize']['range_min'] = wilt_time_min
    default_config['wilting_times']['randomize']['range_max'] = wilt_time_max

    default_config['water_use_efficiencies']['default_value'] = c_water
    default_config['light_use_efficiencies']['default_value'] = c_light
    default_config['x_coordinates']['default_value'] = x_c
    default_config['y_coordinates']['default_value'] = y_c
    if not randomize_coordinates:
        default_config['x_coordinates'].pop('randomize')
        default_config['y_coordinates'].pop('randomize')
    # default_config['garden_width'] = size
    # default_config['garden_length'] = size

    amount_plants = default_config['amount_plants']['default_value']

    cols = default_config['garden_length']['default_value']

    rows = default_config['garden_width']['default_value']

    garden_type = ['real', 'invasive'][is_invasive]
    coords = ['fixed', 'random'][randomize_coordinates]

    path = '/Users/sebastianoehme/fast_ag/gym_fastag/envs/config'
    file = str() + 'plants_' + str(amount_plants) + 'types_' + str(rows) + 'x' + str(
        cols) + 'garden_' + garden_type + '_' + str(coords) + '_test.yaml'

    where_to = os.path.join(path, file)
    print(where_to)
    with open(where_to, 'w') as f:
        data = yaml.safe_dump(default_config, f, sort_keys=False, default_flow_style=None)


if __name__ == '__main__':
    create_config_yaml()
