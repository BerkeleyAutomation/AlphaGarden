import pathlib
import configparser
from simulatorv2 import sim_globals

class FileUtils:
    def __init__(self):
        pass

    def create_config(self, rl_time_steps=3000000, garden_time_steps=40, garden_x=10, garden_y=10, sector_width=2, sector_height=2, num_plant_types=2, num_plants_per_type=1, step=1, action_low=0.0, action_high=sim_globals.MAX_WATER_LEVEL, obs_low=0, obs_high=1000, ent_coef=0.01, n_steps=40000, nminibatches=4, noptepochs=4, learning_rate=1e-8, cnn_args=None):
        config = configparser.ConfigParser()
        config.add_section('rl')
        config['rl']['time_steps'] = str(rl_time_steps)
        config['rl']['ent_coef'] = str(ent_coef)
        config['rl']['n_steps'] = str(n_steps)
        config['rl']['nminibatches'] = str(nminibatches)
        config['rl']['noptepochs'] = str(noptepochs)
        config['rl']['learning_rate'] = str(learning_rate)
        if cnn_args:
            config.add_section('cnn')
            config['cnn']['output_x'] = str(cnn_args["OUTPUT_X"])
            config['cnn']['output_y'] = str(cnn_args["OUTPUT_Y"])
            config['cnn']['num_hidden_layers'] = str(cnn_args["NUM_HIDDEN_LAYERS"])
            config['cnn']['num_filters'] = str(cnn_args["NUM_FILTERS"])
            config['cnn']['num_convs'] = str(cnn_args["NUM_CONVS"])
            config['cnn']['filter_size'] = str(cnn_args["FILTER_SIZE"])
            config['cnn']['stride'] = str(cnn_args["STRIDE"])
            config['cnn']['cc_coef'] = str(cnn_args['CC_COEF'])
            config['cnn']['water_coef'] = str(cnn_args['WATER_COEF'])
        config.add_section('garden')
        config['garden']['time_steps'] = str(garden_time_steps)
        config['garden']['X'] = str(garden_x)
        config['garden']['Y'] = str(garden_y)
        config['garden']['sector_width'] = str(sector_width)
        config['garden']['sector_height'] = str(sector_height) 
        config['garden']['num_plant_types'] = str(num_plant_types)
        config['garden']['num_plants_per_type'] = str(num_plants_per_type)
        config['garden']['step'] = str(step)
        config.add_section('action')
        config['action']['low'] = str(action_low)
        config['action']['high'] = str(action_high)
        config.add_section('obs')
        config['obs']['low'] = str(obs_low)
        config['obs']['high'] = str(obs_high)

        pathlib.Path('gym_config').mkdir(parents=True, exist_ok=True)
        with open('gym_config/config.ini', 'w') as configfile:
            config.write(configfile)

    def createRLSingleRunFolder(self, garden_x, garden_y, num_plant_types, num_plants_per_type, rl, policy_kwargs, time):
        parent_folder = rl['rl_algorithm']
        pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
        sub_folder = parent_folder + '/' + str(garden_x) + 'x' + str(garden_y) + '_garden_' + str(num_plant_types*num_plants_per_type) + '_plants_' + str(rl['time_steps']) + '_timesteps_' + str(rl['learning_rate']) + '_learningrate_' + str(rl['n_steps']) + '_batchsize_' + str(policy_kwargs['CC_COEF']) + '_cropcoef_' + str(policy_kwargs['WATER_COEF']) + '_watercoef_' + time 
        pathlib.Path(sub_folder).mkdir(parents=True, exist_ok=False)
        return sub_folder

    def createBaselineSingleRunFolder(self, garden_x, garden_y, num_plant_types, num_plants_per_type, policy_kwargs, time):
        parent_folder = 'Baseline' 
        pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
        sub_folder = parent_folder + '/' + str(garden_x) + 'x' + str(garden_y) + '_garden_' + str(num_plant_types*num_plants_per_type) + '_plants_' + str(policy_kwargs['CC_COEF']) + '_cropcoef_' + str(policy_kwargs['WATER_COEF']) + '_watercoef_' + time 
        pathlib.Path(sub_folder).mkdir(parents=True, exist_ok=False)
        return sub_folder