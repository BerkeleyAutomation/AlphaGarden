import functools
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import unittest
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from PIL import Image, ImageOps
import os

from scipy.stats import truncnorm
from ray.rllib.env.env_context import EnvContext


class Pest:
    # Future class for modeling pests.
    pass


class Weather:
    # Rain
    # --- amount
    # Sun
    # --- Photosynthetically Active Photon Flux Density (PPFD)
    # Wind
    # --- mass flow
    # Atmosphere
    # --- Humidity
    # --- Temperature
    # --- Air Pressure
    # --- CO2 concentration
    def __init__(self, rows, columns):
        # self.light_flux = 1000
        self.light_flux_grid = np.full((rows, columns), 1000 * 0.01 * 0.01 * 60 * 60, dtype=np.float32)
        self.humidity = 50  # Humidity: ranges in [0,100] %
        self.air_temperature = 21  # Air temperature [°C]

    def reset(self):
        # ToDo: here could be a reset
        # self.light_flux_grid.fill(1000 * 0.01 * 0.01 * 60 * 60)  # PPFD: ~[0,2000] μmol/m^2 s]-->[0,12][μmol/cm^2 min]
        # self.humidity = 50  # Humidity: ranges in [0,100] %
        # self.air_temperature = 21  # Air temperature [°C]
        pass

    def simulation_step(self):
        # ToDo: here could be a model
        pass

    def set_conditions(self, new_light, new_humidity, new_air_temp):
        self.light_flux_grid.fill(new_light)
        self.humidity = new_humidity
        self.air_temperature = new_air_temp


class Soil:
    # Water
    # --- current_amount
    # --- max_water_content
    # --- evaporation
    # Nutrients
    # Oxygen
    # Salinity
    # Structure
    # --- bulk density
    # --- max water holding capacity
    # --- permanent wilting point ?
    # Temperature
    def __init__(self, rows, columns, max_water_content, max_nutrient_content, evaporation_percent_mean,
                 evaporation_percent_scale, init_water_mean, init_water_scale, irr_health_window_width,
                 permanent_wilting_point, np_random):
        self.irr_health_window_width = irr_health_window_width
        self.rows = rows
        self.columns = columns

        self.max_water_content = max_water_content  # max Volumetric Water Content [m^3/m^3]
        self.max_nutrient_content = max_nutrient_content  # constant for grid or cell specific?

        self.evap_dim = Dimension(evaporation_percent_mean, scale=evaporation_percent_scale, range_min=0.0,
                                  range_max=1.0, distribution_type='truncated_normal', shape=(self.rows, self.columns))

        # alternative -0.01675 * ln(dt) for dt = 1,2...,n in 30 min
        self.evaporation = self.evap_dim.default_or_mean_val
        # intervals
        self.permanent_wilting_point = permanent_wilting_point

        self.water_grid_dim = Dimension(init_water_mean, scale=init_water_scale, range_min=0.0,
                                        range_max=self.max_water_content, distribution_type='truncated_normal',
                                        shape=(self.rows, self.columns))
        self.water_grid_dim.randomize(np_random)
        self.water_content_grid = (self.water_grid_dim.current_value).astype(np.float32)

        self.nutrient_content_grid = np.full(shape=(self.rows, self.columns), fill_value=0.0, dtype=np.float32)  # Todo

    def reset(self, np_random):
        """self.water_content_grid = np.clip(np.random.normal(self.init_water_mean, self.init_water_scale,
                                                           (self.rows, self.columns)
                                                           ).astype(np.float32), 0.0, self.max_water_content)"""
        self.water_grid_dim.randomize(np_random)
        self.water_content_grid = self.water_grid_dim.current_value.astype(np.float32)
        # self.water_content_grid = np.full(shape=(self.rows, self.columns), fill_value=self.init_water_mean,
        #                                  dtype=np.float32)
        self.nutrient_content_grid.fill(0.0)  # Todo: randomize

    def calculate_treatment_grid(self, x, y, treatment_actions, x_0, y_0, r_0, gain_slope, amount):
        """Two dimensional irrigation kernel model"""

        r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)  # Circle
        core_range = r <= r_0
        window_grid_size = np.pi * (self.irr_health_window_width ** 2) / 10000  # in square meters
        k = 1.175  # scaling factor to account for water loss from drainage and etc., determined experimentally
        peak_gain = (amount / (window_grid_size * 0.2)) * k  # 1.0
        # peak_gain = 0.05
        # print("PEAK: ", peak_gain)
        outer_gain_range = np.logical_and(r > r_0, r <= r_0 + peak_gain / gain_slope)
        outer_gain = peak_gain + gain_slope * (r_0 - r)
        grid = np.sum(np.select([core_range, outer_gain_range],
                                [peak_gain, outer_gain]) * treatment_actions, axis=0)
        return np.minimum(grid, self.max_water_content)

    def distribute_irrigation(self, irrigation_grid: np.ndarray) -> np.ndarray:
        self.water_content_grid += irrigation_grid
        np.minimum(self.water_content_grid, self.max_water_content, out=self.water_content_grid)
        return np.sum(irrigation_grid)

    def distribute_fertilizer(self, fertilizer_grid: np.ndarray) -> np.ndarray:
        self.nutrient_content_grid += fertilizer_grid
        np.minimum(self.nutrient_content_grid, self.max_nutrient_content, out=self.nutrient_content_grid)
        return np.sum(fertilizer_grid)

    def calculate_grid_contents_balances(self, plants_nutrient_demand_grids: np.ndarray,
                                         plants_water_demand_grids: np.ndarray, np_random: np.random.RandomState
                                         ) -> Tuple[np.ndarray, np.ndarray]:
        # n1 = - kNutSize * sz;
        # n = n + n1 * dt;
        # n = max(0, n);
        total_nutrient_demand_grid = np.sum(plants_nutrient_demand_grids, axis=0)
        total_nutrients_to_adsorb_grid = np.minimum(total_nutrient_demand_grid, self.nutrient_content_grid)

        plants_proportional_nutrients_demand = np.divide(plants_nutrient_demand_grids,
                                                         total_nutrient_demand_grid,
                                                         out=np.zeros_like(plants_nutrient_demand_grids),
                                                         where=total_nutrient_demand_grid != 0)

        plants_nutrient_uptake_grids = total_nutrients_to_adsorb_grid * plants_proportional_nutrients_demand
        self.nutrient_content_grid -= np.sum(plants_nutrient_uptake_grids, axis=0)

        np.maximum(0.0, self.nutrient_content_grid, out=self.nutrient_content_grid)

        # w1 = - kWaterSize*sz - kWaterEvap*w
        # water_to_absorb_grid = min ( water_grid_content, plants_desired_water_amount_grid)
        # sync water_to_absorb with plants
        # w1 = - water_to_absorb_grid  - evaporation_parameter * water_grid
        # w = w + w1*dt
        # TODO - change evaporation depending on day last watered
        self.evap_dim.randomize(np_random)
        self.evaporation = self.evap_dim.current_value
        self.water_content_grid -= self.evaporation * self.water_content_grid

        total_water_demand_grid = np.sum(plants_water_demand_grids, axis=0)
        total_water_to_adsorb_grid = np.clip(total_water_demand_grid, 0,
                                             self.water_content_grid - self.permanent_wilting_point)

        plants_proportional_water_demand = np.divide(plants_water_demand_grids,
                                                     total_water_demand_grid,
                                                     out=np.zeros_like(plants_water_demand_grids),
                                                     where=total_water_demand_grid != 0)

        plants_water_uptake_grids = total_water_to_adsorb_grid * plants_proportional_water_demand

        self.water_content_grid -= np.sum(plants_water_uptake_grids, axis=0) * (1 / 4)

        np.maximum(0.0, self.water_content_grid, out=self.water_content_grid)

        return plants_nutrient_uptake_grids, plants_water_uptake_grids


class Plants:
    # Morphology
    # --- Shoot system
    # ------- Stems
    # ------- Leaves
    # ------- Flower / Fruit
    # --- Root system
    # ------ Primary root
    # ------ Secondary roots

    # chronological age (calendar age) vs ontogenetic age (elapsed time after seed germination)
    def __init__(self, rows: int, columns: int, xv: np.ndarray, yv: np.ndarray, amount_plants: int, amount_plant_types,
                 overwatered_time_threshold: int, underwatered_time_threshold: int, overwater_threshold: float,
                 underwater_threshold: float, reference_outer_radii: np.ndarray,
                 common_names: List[str], germination_times: np.ndarray, maturation_times: np.ndarray,
                 waiting_stage_durations: np.ndarray, wilting_stage_durations: np.ndarray,
                 water_use_efficiencies: np.ndarray, light_use_efficiencies: np.ndarray,
                 nutrients_use_efficiencies: np.ndarray, x_coordinates: np.ndarray,
                 y_coordinates: np.ndarray):

        self.amount_plants = amount_plants
        self.amount_plant_types = amount_plant_types
        self.wilting_factors = np.zeros(self.amount_plants)
        self.ids = np.arange(self.amount_plants)
        self.common_names = common_names
        _, self.plant_type_ids = np.unique(self.common_names, return_inverse=True)
        self.xv = xv
        self.yv = yv

        # self.chronological_ages = np.empty(AMOUNT_PLANTS, dtype=np.int)  #

        self.germination_stage_durations = germination_times.astype(np.int)
        self.maturation_durations = maturation_times.astype(np.int)
        self.growth_stage_durations = self.maturation_durations - self.germination_stage_durations
        self.waiting_stage_durations = waiting_stage_durations.astype(np.int)
        self.wilting_stage_durations = wilting_stage_durations.astype(np.int)

        # self.death_time_mean = np.full(self.amount_plants, -1, dtype=np.float32)
        # self.death_time_scale = np.full(self.amount_plants, 0, dtype=np.float32)
        self.death_stage_durations = np.full(self.amount_plants, -1, dtype=np.int)

        self.overwatered_flags = np.full(self.amount_plants, False, dtype=np.bool)
        self.underwatered_flags = np.full(self.amount_plants, False, dtype=np.bool)
        self.stress_times = np.zeros(self.amount_plants, dtype=np.int)

        self.underwatered_time_threshold = underwatered_time_threshold
        self.overwatered_time_threshold = overwatered_time_threshold
        self.underwater_threshold = underwater_threshold
        self.overwater_threshold = overwater_threshold

        self.current_outer_radii = np.zeros(self.amount_plants, dtype=np.float32)
        self.current_heights = np.zeros(self.amount_plants, dtype=np.float32)
        self.current_structure = np.zeros(self.amount_plants, dtype=np.float32)

        self.current_total_desired_transpiration = np.zeros((self.amount_plants, 1, 1), dtype=np.float32)
        self.current_total_desired_nutrients = np.zeros((self.amount_plants, 1, 1), dtype=np.float32)

        self.current_lai_grids = np.zeros((self.amount_plants, rows, columns), dtype=np.float32)  # []
        self.current_grid_locations = np.full((self.amount_plants, rows, columns), False, dtype=bool)

        self.unoccluded_ratio = np.full(self.amount_plants, 1.0, dtype=np.float32)

        # dW = LUE * PPFD * (1 - exp(-k LAI)) = Yg ( Pg - Rm ) = LUE * PAR - LUE * PAR * exp(-k LAI)
        # Rm = maintenance coef * Dry Biomass Weight
        # Pg is gross photosynthesis
        # Yg is growth conversion efficiency

        self.current_light_exposures = np.zeros(self.amount_plants, dtype=np.float32)  # (sun) light energy uptake
        self.current_growth_stages = np.zeros(self.amount_plants, dtype=np.int)
        self.current_remaining_stage_times = self.calculate_new_stage_time(self.current_growth_stages, self.ids)
        self.current_health_status = np.full(self.amount_plants, 2, dtype=np.int)

        # add colors: as matrix or single vectors?
        # add color_step
        # add stop_colors

        # companionship_factor

        self.x_coordinates = x_coordinates.reshape((self.amount_plants, 1, 1))
        self.y_coordinates = y_coordinates.reshape((self.amount_plants, 1, 1))
        self.reference_outer_radii = reference_outer_radii  # Constant?
        self.reference_heights = np.full(self.amount_plants, 0.4, dtype=np.float32)
        self.reference_structure = self.reference_heights * np.pi * self.reference_outer_radii ** 2  # Constant?

        self.cross_section_coefficients = np.full((self.amount_plants, 1, 1), 0.7, dtype=np.float32)  # Constant
        self.water_use_efficiencies = water_use_efficiencies  # Constant?
        self.light_use_efficiencies = light_use_efficiencies  # Constant?
        self.nutrients_use_efficiencies = water_use_efficiencies  # TODO: FIX to real
        self.k1_values = np.full(self.amount_plants, 0.3, dtype=np.float32)  # Constant?
        self.k2_values = np.full(self.amount_plants, 0.7, dtype=np.float32)  # Constant?
        # self.permanent_wilting_points init  # Constant? #

        self.overwatered_wilting_factor = 0.8 ** (1 / self.overwatered_time_threshold)
        self.underwatered_wilting_factor = 0.8 ** (1 / self.underwatered_time_threshold)

        # self.light_compensation_points = np.empty(self.amount_plants, dtype=np.float32)
        # photosynthesis=respiration[μmol]

    def calculate_new_stage_time(self, stage_indices, unique_ids):

        condlist = [stage_indices == 0,
                    stage_indices == 1,
                    stage_indices == 2,
                    stage_indices == 3]

        choicelist = [self.germination_stage_durations[unique_ids],
                      self.growth_stage_durations[unique_ids],
                      self.waiting_stage_durations[unique_ids],
                      self.wilting_stage_durations[unique_ids]]
        return np.select(condlist, choicelist, default=-1)

    def reset(self):

        self.current_outer_radii.fill(0.0)
        self.current_heights.fill(0.0)
        self.current_structure.fill(0.0)

        self.overwatered_flags.fill(False)
        self.underwatered_flags.fill(False)
        self.stress_times.fill(0)

        self.current_total_desired_transpiration.fill(0.0)
        self.current_total_desired_nutrients.fill(0.0)

        # self.current_lai_grids = self.update_leaf_area_index_grids(self.xv, self.yv, 1.0,
        #                                                            self.x_coordinates, self.y_coordinates,
        #                                                            self.current_outer_radii.reshape(
        #                                                            self.amount_plants, 1,1))
        # self.current_grid_locations = self.current_lai_grids > 0
        self.current_lai_grids.fill(0.0)
        self.current_grid_locations.fill(False)

        self.unoccluded_ratio.fill(1.0)

        self.current_light_exposures.fill(0.0)  # (sun) light energy uptake
        self.current_growth_stages.fill(0)

        # TODO add time update function
        self.current_remaining_stage_times = self.calculate_new_stage_time(self.current_growth_stages, self.ids)
        self.current_health_status.fill(2)

        # self.light_compensation_points

    def get_health_status(self):
        """ Update heath status values depending on plant stage and environment condition. """
        germination_stage_flags = self.current_growth_stages == 0
        growth_or_waiting_flags = (self.current_growth_stages == 1) | (self.current_growth_stages == 2)
        under_g_w_flags = self.underwatered_flags & growth_or_waiting_flags
        over_g_w_flas = self.overwatered_flags & growth_or_waiting_flags
        death_or_wilting_stage_flags = (self.current_growth_stages == 3) | (self.current_growth_stages == 4)

        hs_normal = np.full(self.amount_plants, 2, dtype=np.int)  # germinating, growing, waiting normal
        hs_overwater = np.full(self.amount_plants, 3, dtype=np.int)  # growing, waiting overwater
        hs_underwater = np.ones(self.amount_plants, dtype=np.int)  # growing, waiting underwater
        hs_death_wilting = np.zeros(self.amount_plants, dtype=np.int)  # no plant, dead, wilting

        cond_list = [germination_stage_flags, growth_or_waiting_flags, under_g_w_flags, over_g_w_flas,
                     death_or_wilting_stage_flags]

        choice_list = [hs_normal, hs_normal, hs_underwater, hs_overwater, hs_death_wilting]

        return np.select(cond_list, choice_list, default=-1)

    def get_circular_lai_grids(self, x, y, leaf_area_indices, centers_x, centers_y, outer_radii):
        """Two dimensional Disk model function"""

        rr = np.sqrt((x - centers_x) ** 2 + (y - centers_y) ** 2)
        return np.select([rr < outer_radii.reshape(self.amount_plants, 1, 1)], [leaf_area_indices])

    def get_sorted_descending_heights_indices(self):
        """ Calculate index array for current sorted descending plant heights. """
        return self.current_heights.argsort()[::-1]

    def calculate_desired_uptake_grids(self, light_flux_grid: np.ndarray):
        idx = self.get_sorted_descending_heights_indices()  # updates idx
        """
        # calculate grids with height dependant chained attenuation measure of the transmitted radiant power in plants.
        cumulative_transmittance_grids = np.cumsum(
            self.cross_section_coefficients[idx] * self.current_lai_grids[idx], axis=0)[self.ids[idx].argsort()]
        # calculate individual transmittance grids of plants
        individual_transmittance_grids = self.cross_section_coefficients * self.current_lai_grids
        # Update the current total light exposures
        self.current_light_exposures = np.sum(
            light_flux_grid * np.exp(- (cumulative_transmittance_grids - individual_transmittance_grids)) * (
                    1 - np.exp(- individual_transmittance_grids)), axis=(1, 2))
        max_light_exposure = np.sum(light_flux_grid * (1 - np.exp(- individual_transmittance_grids)), axis=(1, 2))
        self.unoccluded_ratio = np.divide(self.current_light_exposures,
                                          max_light_exposure,
                                          out=np.zeros_like(max_light_exposure),
                                          where=max_light_exposure != 0).astype(np.float32)
        """
        # Cumulative Leaf Area Index (LAI) tensor (self.amount_plants, rows, columns)
        cumulative_lai = np.cumsum(self.current_lai_grids[idx], axis=0)[self.ids[idx].argsort()]
        self.current_grid_locations = self.current_lai_grids > 0

        # Total current light exposure vector for each plant
        self.current_light_exposures = np.sum(self.current_grid_locations * np.select(
            [cumulative_lai == 1, cumulative_lai == 2, cumulative_lai == 3], [1, 0.5, 0.25], 0), axis=(1, 2))

        # Total grid area vector for each plant
        current_total_grid_points = np.count_nonzero(self.current_lai_grids, axis=(1, 2)).astype(
            np.float32)  # .reshape(self.amount_plants, 1, 1)

        # Calculate unoccluded ratio vector with fraction of plants current light exposure over plant grid area
        self.unoccluded_ratio = np.divide(self.current_light_exposures,
                                          current_total_grid_points,
                                          out=np.zeros_like(current_total_grid_points),
                                          where=current_total_grid_points != 0).astype(np.float32)

        # Update plants max possible productivity
        maximal_productivity = self.light_use_efficiencies * (self.current_light_exposures ** 0.5)

        # Update wilting stage dependant discounted productivity
        maximal_productivity[self.current_growth_stages == 3] = (
                (self.current_remaining_stage_times / self.wilting_stage_durations) * maximal_productivity
        )[self.current_growth_stages == 3]

        # Update death stage dependant productivity
        maximal_productivity[self.current_growth_stages == 4] = 0

        # Update total desired water
        self.current_total_desired_transpiration = (maximal_productivity).reshape(
            self.amount_plants, 1, 1)
        # self.water_use_efficiencies).reshape(self.amount_plants, 1, 1)

        # Update total desired nutrients
        self.current_total_desired_nutrients = (maximal_productivity / self.nutrients_use_efficiencies).reshape(
            self.amount_plants, 1, 1)  # Todo check c3m

        # Reshape current total plant grid point area vector for broadcasting calculation
        current_total_grid_points = current_total_grid_points.reshape((self.amount_plants, 1, 1))

        # Calculate current desired water uptake grids
        current_desired_water_uptake_grids = self.current_grid_locations * np.divide(
            self.current_total_desired_transpiration, current_total_grid_points,
            out=np.zeros_like(self.current_lai_grids), where=current_total_grid_points != 0)

        # Todo update with nutrients model; Calculate current desired nutrient uptake grid
        current_desired_nutrient_uptake_grids = self.current_grid_locations * np.divide(
            self.current_total_desired_nutrients, current_total_grid_points,
            out=np.zeros_like(self.current_lai_grids), where=current_total_grid_points != 0)
        return current_desired_nutrient_uptake_grids, current_desired_water_uptake_grids

    def grow(self, plants_nutrient_uptake_grids: np.ndarray, plants_water_uptake_grids: np.ndarray,
             past_water_demand_grids: np.ndarray, past_nutrients_demand_grids: np.ndarray,
             past_total_available_water: np.ndarray):
        plants_water_contents = np.sum(plants_water_uptake_grids, axis=(1, 2))
        past_water_demand = np.sum(past_water_demand_grids, axis=(1, 2))
        # plants_nutrient_contents = np.sum(plants_nutrient_uptake_grids, axis=(1, 2))
        # past_nutrient_demand = np.sum(past_nutrients_demand_grids, axis=(1, 2))

        upward_growth = np.zeros(self.amount_plants)
        outward_growth = np.zeros(self.amount_plants)

        updated_flag = np.zeros(self.amount_plants, dtype=np.bool)

        germination_stage_flags = self.current_growth_stages == 0
        growth_stage_flags = self.current_growth_stages == 1
        waiting_stage_flags = self.current_growth_stages == 2
        wilting_stage_flags = self.current_growth_stages == 3
        death_stage_flags = self.current_growth_stages == 4

        growth_or_waiting_stage_flag = growth_stage_flags | waiting_stage_flags

        overwatered_condition = past_total_available_water > self.overwater_threshold * past_water_demand
        overwater_reset_condition = ~overwatered_condition & self.overwatered_flags & growth_or_waiting_stage_flag
        self.stress_times[overwater_reset_condition] = 0
        upward_growth[overwater_reset_condition] = 0
        outward_growth[overwater_reset_condition] = 0
        updated_flag += overwater_reset_condition

        over_watered_growth_waiting = overwatered_condition & growth_or_waiting_stage_flag
        self.stress_times[over_watered_growth_waiting] += 1
        self.overwatered_flags[growth_or_waiting_stage_flag] = overwatered_condition[
            growth_or_waiting_stage_flag]
        upward_growth[over_watered_growth_waiting] = 0
        outward_growth[over_watered_growth_waiting] = (self.overwatered_wilting_factor - 1
                                                       ) * self.current_outer_radii[over_watered_growth_waiting]
        updated_flag += over_watered_growth_waiting

        underwatered_condition = plants_water_contents < self.underwater_threshold * past_water_demand
        underwater_reset_condition = ~underwatered_condition & self.underwatered_flags & growth_or_waiting_stage_flag
        self.stress_times[underwater_reset_condition] = 0
        upward_growth[underwater_reset_condition] = 0
        outward_growth[underwater_reset_condition] = 0
        updated_flag += underwater_reset_condition

        under_watered_growth_waiting = underwatered_condition & growth_or_waiting_stage_flag

        self.stress_times[under_watered_growth_waiting] += 1
        self.underwatered_flags[growth_or_waiting_stage_flag] = underwatered_condition[
            growth_or_waiting_stage_flag]
        upward_growth[under_watered_growth_waiting] = 0
        outward_growth[under_watered_growth_waiting] = (self.overwatered_wilting_factor - 1
                                                        ) * self.current_outer_radii[under_watered_growth_waiting]
        updated_flag += under_watered_growth_waiting

        # maintenance = (self.current_structure / self.reference_structure) * (1 / self.current_health_status)
        wis_flag = wilting_stage_flags & ~updated_flag
        upward_growth[wis_flag] = 0
        outward_growth[wis_flag] = (self.wilting_factors[wis_flag] - 1) * self.current_outer_radii[wis_flag]
        updated_flag += wilting_stage_flags

        maintenance = self.current_outer_radii / self.reference_outer_radii
        maintenance = np.minimum(maintenance, 1.0)

        # growth_pool = np.minimum(self.water_use_efficiencies * plants_water_contents,
        #                         self.nutrients_use_efficiencies * plants_nutrient_contents
        #                         ) * (1 - maintenance)  #TODO fix
        growth_pool = self.water_use_efficiencies * plants_water_contents * (1 - maintenance)
        radial_proportion = np.minimum(np.maximum(self.k1_values, self.unoccluded_ratio), self.k2_values)
        gs_flag = growth_stage_flags & ~updated_flag
        outward_growth[gs_flag] = (radial_proportion * growth_pool)[gs_flag]
        upward_growth[gs_flag] = ((1 - radial_proportion) * growth_pool)[gs_flag]
        self.update_structure(upward=upward_growth, outward=outward_growth)

    def prune(self, prune_actions, prune_rate):
        """ """
        amount_to_prune = prune_rate * self.current_outer_radii * prune_actions  # TODO refactor prune_rate: policy
        self.update_structure(outward=-amount_to_prune)
        self.current_lai_grids = self.get_circular_lai_grids(self.xv, self.yv, 1.0,
                                                             self.x_coordinates, self.y_coordinates,
                                                             self.current_outer_radii.reshape(
                                                                 (self.amount_plants, 1, 1)))

    def update_structure(self, upward=None, outward=None):
        """Update plant size after growth, stress or pruning.
        Args:
            upward (np.ndarray): new vertical size update
            outward (np.ndarray): new horizontal size update
        """

        if upward is not None:
            self.current_heights += upward
        if outward is not None:
            self.current_outer_radii += outward
            np.clip(self.current_outer_radii, 0.00, self.reference_outer_radii, out=self.current_outer_radii)
        self.current_structure = self.current_heights * np.pi * self.current_outer_radii ** 2


class WeatherTestCase(unittest.TestCase):
    pass


class SoilTestCase(unittest.TestCase):
    pass


class PlantTestCase(unittest.TestCase):
    pass


class Dimension(object):
    """Class which handles the machinery of randomizing a particular dimension
    """

    def __init__(self, default_value, multiplier_min=None, multiplier_max=None,
                 scale=None, range_min=None, range_max=None, distribution_type=None, shape=None):
        self.distribution_type = distribution_type
        self.default_or_mean_val = default_value
        self.current_value = default_value
        self.shape = shape

        if distribution_type == 'uniform':
            self.range_min = self.default_or_mean_val * multiplier_min
            self.range_max = self.default_or_mean_val * multiplier_max
        elif self.distribution_type == 'truncated_normal':
            self.scale_value = scale
            assert np.any((self.default_or_mean_val > range_min) & (self.default_or_mean_val < range_max))
            self.range_min = range_min
            self.range_max = range_max
            self.scipy_truncnorm = truncnorm
            # calculate alpha and beta for mean-preserving contraction

            self.alpha = np.divide((self.range_min - self.default_or_mean_val), self.scale_value,
                                   out=np.copy(self.range_min), where=self.scale_value != 0, casting='unsafe')
            self.beta = np.divide((self.range_max - self.default_or_mean_val), self.scale_value,
                                  out=np.copy(self.range_max), where=self.scale_value != 0, casting='unsafe')
        elif self.distribution_type == 'random_coordinate':
            self.range_min = int(range_min)
            self.range_max = int(range_max)
        elif distribution_type is None:
            pass
        else:
            raise ValueError('No distribution type called {}'.format(self.distribution_type))

    def randomize(self, np_random):
        if self.distribution_type == 'uniform':
            self.current_value = np_random.uniform(low=self.range_min,
                                                   high=self.range_max, size=self.shape)  # Check if data type needed
        elif self.distribution_type == 'truncated_normal':
            self.current_value = self.scipy_truncnorm.rvs(self.alpha, self.beta, loc=self.default_or_mean_val,
                                                          scale=self.scale_value, size=self.shape,
                                                          random_state=np_random)
        elif self.distribution_type == 'random_coordinate':
            self.current_value = (np_random.randint(self.range_min, high=self.range_max,
                                                    size=self.shape) / 100).astype(np.float32)

        else:
            raise ValueError('No distribution type called {}'.format(self.distribution_type))

    def reset(self):
        self.current_value = self.default_or_mean_val

    def set(self, value):
        self.current_value = value


# Process-based model for poly (culture) crop garden
# according to endogenous plant properties and environmental conditions.
class FastAg(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config):
        super(FastAg, self).__init__()

        self.worker_index = 0
        self.vector_index = 0
        self.trail_seed = 0
        self.num_workers = 1
        self.validation_seed = 10000 if env_config.get('is_validation_seed') else 0

        if isinstance(env_config, EnvContext):
            self.worker_index = env_config.worker_index
            self.vector_index = env_config.vector_index
            self.num_workers = env_config.num_workers

        self.dimensions = {}

        self.np_random = None
        # estimate to have max 100 envs per worker
        self.seed(self.worker_index * 100 + self.vector_index + self.trail_seed + self.validation_seed)

        self.current_day = 0
        self.day_limit = env_config['garden_days']['default_value']  # 365? # = Episode length?

        # Garden Parameters
        self.garden_length = env_config['garden_length']['default_value']
        self.garden_width = env_config['garden_width']['default_value']

        self.sector_rows = env_config['sector_rows']['default_value']
        self.sector_cols = env_config['sector_cols']['default_value']
        # self.cell_size = env_config['cell_size']['default_value']
        self.rows = self.garden_width  # number of grid rows
        self.columns = self.garden_length  # number of grid columns
        self.yv, self.xv = np.meshgrid(np.linspace(0.0, self.garden_width, self.rows, endpoint=False),
                                       np.linspace(0.0, self.garden_length, self.columns, endpoint=False),
                                       indexing='ij')

        self.max_water_content = self.check_setup(env_config, 'max_water_content')

        self.water_saturation_thr = 0.6  # highest known water volumetric water content (VWC) for soil

        self.max_nutrient_content = self.check_setup(env_config, 'max_nutrient_content')
        self.permanent_wilting_point = self.check_setup(env_config, 'permanent_wilting_point')
        self.init_water_mean = env_config['init_water_mean']['default_value']
        self.init_water_scale = env_config['init_water_scale']['default_value']
        self.evaporation_percent_mean = env_config['evaporation_percent_mean']['default_value']  # NEW || old --> 0.06
        self.evaporation_percent_scale = env_config['evaporation_percent_scale']['default_value']  # NEW
        self.irrigation_amount = env_config['irrigation_amount']['default_value']  # NEW || 200mL in cubic metre

        # Policy Parameters
        self.water_threshold = env_config['water_threshold']['default_value']
        self.irr_health_window_width = env_config['irr_health_window_width']['default_value']
        self.prune_window_rows = env_config['prune_window_rows']['default_value']
        self.prune_window_cols = env_config['prune_window_cols']['default_value']
        self.prune_rate = env_config['prune_rate']['default_value']
        self.prune_delay = env_config['prune_delay']['default_value']

        # Plant Parameters
        self.amount_plants = env_config['amount_plants']['default_value']
        self.amount_plant_types = env_config['amount_plant_types']['default_value']
        self.common_names = env_config['common_names']['default_value']
        self.x_coordinates = self.check_setup(env_config, 'x_coordinates')
        self.y_coordinates = self.check_setup(env_config, 'y_coordinates')
        self.reference_outer_radii = self.check_setup(env_config, 'reference_outer_radii')
        self.light_use_efficiencies = env_config['light_use_efficiencies']['default_value']
        self.water_use_efficiencies = self.check_setup(env_config, 'water_use_efficiencies')
        self.nutrients_use_efficiencies = env_config['nutrients_use_efficiencies']['default_value']

        self.germination_times = self.check_setup(env_config, 'germination_times')

        self.maturation_times = self.check_setup(env_config, 'maturation_times')

        self.waiting_times = self.check_setup(env_config, 'waiting_times')

        self.wilting_times = self.check_setup(env_config, 'wilting_times')

        self.tau = env_config['tau']['default_value']  # 48 hours
        self.overwatered_time_threshold = env_config['overwatered_time_threshold']['default_value']
        self.underwatered_time_threshold = env_config['underwatered_time_threshold']['default_value']
        self.overwatered_threshold = env_config['overwatered_threshold']['default_value']
        self.underwaterd_threshold = env_config['underwaterd_threshold']['default_value']

        self.weather = Weather(rows=self.rows, columns=self.columns)
        self.soil = Soil(rows=self.rows, columns=self.columns,
                         max_water_content=self.max_water_content, max_nutrient_content=self.max_nutrient_content,
                         evaporation_percent_mean=self.evaporation_percent_mean,
                         evaporation_percent_scale=self.evaporation_percent_scale,
                         init_water_mean=self.init_water_mean, init_water_scale=self.init_water_scale,
                         irr_health_window_width=self.irr_health_window_width,
                         permanent_wilting_point=self.permanent_wilting_point,
                         np_random=self.np_random)

        self.plants = Plants(rows=self.rows, columns=self.columns, xv=self.xv, yv=self.yv,
                             amount_plants=self.amount_plants, amount_plant_types=self.amount_plant_types,
                             overwatered_time_threshold=self.overwatered_time_threshold,
                             underwatered_time_threshold=self.underwatered_time_threshold,
                             overwater_threshold=self.overwatered_threshold,
                             underwater_threshold=self.underwaterd_threshold,
                             reference_outer_radii=self.reference_outer_radii, common_names=self.common_names,
                             germination_times=self.germination_times,
                             maturation_times=self.maturation_times,
                             waiting_stage_durations=self.waiting_times,
                             wilting_stage_durations=self.wilting_times,
                             water_use_efficiencies=self.water_use_efficiencies,
                             light_use_efficiencies=self.light_use_efficiencies,
                             nutrients_use_efficiencies=self.nutrients_use_efficiencies,
                             x_coordinates=self.x_coordinates, y_coordinates=self.y_coordinates)

        self.water_uptake_queue = [None for _ in range(self.tau)]
        # [[None for _ in range(self.tau[i])] for i in range(AMOUNT_PLANTS)]
        self.nutrients_uptake_queue = [None for _ in range(self.tau)]
        self.available_water_queue = [None for _ in range(self.tau)]
        self.desired_water_queue = [None for _ in range(self.tau)]
        self.desired_nutrients_queue = [None for _ in range(self.tau)]

        self.current_irrigation_total = 0.0
        self.current_fertilizer_total = 0.0
        self.coverages = []
        self.diversities = []
        self.irrigation_amounts = []
        self.irrigation = 0
        self.total_irrigation = 0
        self.overwater_deaths = 0
        self.underwater_deaths = 0

        """self.observation_space = spaces.Dict({"water_grid": spaces.Box(low=0.0, high=MAX_WATER_CONTENT,
                                                                       shape=(self.garden.rows, self.garden.columns),
                                                                       dtype=np.float32),
                                              "plants_unoccluded_ratios": spaces.Box(low=0.0, high=1.0,
                                                                                     shape=(AMOUNT_PLANTS,),
                                                                                     dtype=np.float32),
                                              "plants_health": spaces.Box(low=0.0, high=1.0,
                                                                          shape=(AMOUNT_PLANTS,),
                                                                          dtype=np.float32),
                                              "plants_structure": spaces.Box(low=0.0, high=1.0, shape=(AMOUNT_PLANTS,),
                                                                             dtype=np.float32)})"""

        self.observation_space = spaces.Dict(
            {"plant_cc_grid": spaces.Box(low=0, high=self.amount_plants + 1,  # with soil +1
                                         shape=(self.rows, self.columns),
                                         dtype=np.int),
             "water_grid": spaces.Box(low=0.0, high=self.max_water_content,  # max_water_content / water_saturation_thr
                                      shape=(self.rows, self.columns),
                                      dtype=np.float32),
             "plants_health": spaces.Box(low=0, high=3,
                                         shape=(self.rows, self.columns),
                                         dtype=np.int),
             })

        """self.observation_space = spaces.Dict(
            {"norm_cc_vec": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plant_types + 1,)),
             "water_grid": spaces.Box(low=0.0, high=self.max_water_content, shape=(self.rows, self.columns),
                                     dtype=np.float32),
             })"""

        """self.action_space = spaces.Dict({"irrigation": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants,)),
                                         "prune": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants,)),
                                         "nutrients": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants,))})"""

        """self.action_space = spaces.Dict({"irrigation": spaces.MultiDiscrete(np.full(self.amount_plants, 2,
                                                                                    dtype=np.int8)),
                                         "prune": spaces.MultiDiscrete(np.full(self.amount_plants, 2,
                                                                               dtype=np.int8)),
                                         "nutrients": spaces.MultiDiscrete(np.full(self.amount_plants, 2,
                                                                                   dtype=np.int8))})"""

        self.action_space = spaces.Dict({
            "irrigation": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants,), dtype=np.int),
            "prune": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants,), dtype=np.int),
            "nutrients": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants,), dtype=np.int)
        })

    def check_setup(self, conf_dict: dict, para_name: str):
        para_dict = conf_dict[para_name]
        val = para_dict['default_value']
        array_dtype = para_dict.get("dtype")
        if array_dtype == 'np.float32':
            val = np.array(val, dtype=np.float32)
        elif array_dtype == 'np.int':
            val = np.array(val, dtype=np.int)
        elif array_dtype == None:
            pass
        else:
            raise ValueError('No such data type enabled {}'.format(array_dtype))
        randomize_dict = para_dict.get('randomize')
        shape = None  # ??
        if randomize_dict:
            if randomize_dict['distribution_type'] == 'uniform':
                dim = Dimension(default_value=val,
                                multiplier_min=np.array(randomize_dict['multiplier_min']),
                                multiplier_max=np.array(randomize_dict['multiplier_max']),
                                distribution_type='uniform',
                                shape=shape
                                )
            elif randomize_dict['distribution_type'] == 'truncated_normal':
                dim = Dimension(default_value=val,
                                scale=np.array(randomize_dict['scale']),
                                range_min=np.array(randomize_dict['range_min']),
                                range_max=np.array(randomize_dict['range_max']),
                                distribution_type='truncated_normal',
                                shape=shape
                                )
            elif randomize_dict['distribution_type'] == 'random_coordinate':
                dim = Dimension(default_value=val,
                                range_min=randomize_dict['range_min'] * 100,
                                range_max=randomize_dict['range_max'] * 100,
                                distribution_type='random_coordinate',
                                shape=randomize_dict['shape']
                                )
            else:
                raise ValueError('No distribution type called {}'.format(randomize_dict['distribution_type']))
            self.dimensions[para_name] = dim
            dim.randomize(self.np_random)
            val = dim.current_value
        return val

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def enqueue(self, queue, data):
        queue.insert(0, data)

    # Removing the front element from the queue
    def dequeue(self, queue):
        if len(queue) > 0:
            return queue.pop()
        else:
            raise Exception('Queue Empty!')

    def simulation_step(self, irrigation_grid: np.ndarray, fertilizer_grid: np.ndarray, prune_actions: np.ndarray):
        self.current_irrigation_total = self.soil.distribute_irrigation(irrigation_grid)  # apply action
        self.current_fertilizer_total = self.soil.distribute_fertilizer(fertilizer_grid)  # apply action
        self.plants.prune(prune_actions, self.prune_rate)
        # Future: weather.distribute_light()?
        # Future: self.weather.simulation_step()

        plants_nutrient_demand_grids, plants_water_demand_grids = self.plants.calculate_desired_uptake_grids(
            self.weather.light_flux_grid)

        # delay water and nutrient demand
        self.enqueue(self.desired_water_queue, plants_water_demand_grids)
        self.enqueue(self.desired_nutrients_queue, plants_nutrient_demand_grids)

        total_available_water_plants = np.sum(self.plants.current_grid_locations * self.soil.water_content_grid,
                                              axis=(1, 2))

        # delay total available soil water to plants
        self.enqueue(self.available_water_queue, total_available_water_plants)

        plants_nutrient_uptake_grids, plants_water_uptake_grids = self.soil.calculate_grid_contents_balances(
            plants_nutrient_demand_grids,
            plants_water_demand_grids, self.np_random)  # Todo time normalization

        # delay water and nutrient uptake
        self.enqueue(self.water_uptake_queue, plants_water_uptake_grids)  # TODO save slices to indiv. queues
        self.enqueue(self.nutrients_uptake_queue, plants_nutrient_uptake_grids)  # TODO save slices to indiv. queues

        # retrieve past total available soil water to plants
        past_total_available_water = self.dequeue(self.available_water_queue)
        # retrieve past water and nutrient demand
        past_plants_water_demand_grids = self.dequeue(self.desired_water_queue)
        past_plants_nutrient_demand_grids = self.dequeue(self.desired_nutrients_queue)
        # retrieve past water and nutrient uptake
        delayed_plants_water_uptake_grids = self.dequeue(self.water_uptake_queue)
        delayed_plants_nutrient_uptake_grids = self.dequeue(self.nutrients_uptake_queue)

        # Transform None queue items to zeros numpy array
        if past_total_available_water is None:
            past_total_available_water = np.zeros_like(self.plants.ids, dtype=np.float32)
        if past_plants_water_demand_grids is None:
            past_plants_water_demand_grids = np.zeros_like(plants_water_demand_grids)
        if past_plants_nutrient_demand_grids is None:
            past_plants_nutrient_demand_grids = np.zeros_like(plants_nutrient_demand_grids)
        if delayed_plants_water_uptake_grids is None:
            delayed_plants_water_uptake_grids = np.zeros_like(plants_water_uptake_grids)
        if delayed_plants_nutrient_uptake_grids is None:
            delayed_plants_nutrient_uptake_grids = np.zeros_like(plants_nutrient_demand_grids)

        # Grow plants according to growth model
        self.plants.grow(delayed_plants_nutrient_uptake_grids, delayed_plants_water_uptake_grids,
                         past_plants_water_demand_grids, past_plants_nutrient_demand_grids,
                         past_total_available_water)

        # Update over watered death condition
        overwatered_death_condition = np.logical_and(self.plants.overwatered_flags,
                                                     self.plants.stress_times >= self.plants.overwatered_time_threshold)
        # Update under watered death condition
        underwatered_death_condition = np.logical_and(self.plants.underwatered_flags,
                                                      self.plants.stress_times >= self.plants.underwatered_time_threshold)

        # Helper console output for dead plants.
        self.overwater_deaths = np.sum(overwatered_death_condition)
        self.underwater_deaths = np.sum(underwatered_death_condition)
        # if np.any(overwatered_death_condition):
        #    print('death overwater')
        # if np.any(underwatered_death_condition):
        #    print('death underwater')
        # check stress dependant stage switch
        self.plants.current_growth_stages[overwatered_death_condition] = 4
        self.plants.stress_times[overwatered_death_condition] = 0
        self.plants.overwatered_flags[overwatered_death_condition] = False

        self.plants.current_growth_stages[underwatered_death_condition] = 4
        self.plants.stress_times[underwatered_death_condition] = 0
        self.plants.underwatered_flags[underwatered_death_condition] = False

        # Check update condition for stage change
        update_condition = np.logical_and(self.plants.current_remaining_stage_times == 0,
                                          self.plants.current_growth_stages < 4)
        # Get new stage index after stage change
        updated_stage_idx_slice = self.plants.current_growth_stages[update_condition] + 1
        # Get plant id of updated plants
        uid_slices = self.plants.ids[update_condition]

        # Check if plants germinate and update structure
        uid_germinated = uid_slices[updated_stage_idx_slice == 1]
        self.plants.current_outer_radii[uid_germinated] = 1
        self.plants.current_heights[uid_germinated] = 1
        self.plants.current_structure[uid_germinated] = 1 * np.pi * 1 ** 2

        # Check if plant is dead
        uid_dead = uid_slices[updated_stage_idx_slice == 4]
        self.plants.current_outer_radii[uid_dead] = 0
        self.plants.current_heights[uid_dead] = 0
        self.plants.current_structure[uid_dead] = 0

        # Update Leaf Area Index grids
        self.plants.current_lai_grids = self.plants.get_circular_lai_grids(self.xv, self.yv, 1.0,
                                                                           self.plants.x_coordinates,
                                                                           self.plants.y_coordinates,
                                                                           self.plants.current_outer_radii.reshape(
                                                                               (self.amount_plants, 1, 1)))
        # Check wilting condition
        uid_wilting = uid_slices[updated_stage_idx_slice == 3]
        self.plants.wilting_factors[uid_wilting] = (2 / (self.plants.current_outer_radii[uid_wilting] + 1e-10)
                                                    ) ** (1 / self.plants.wilting_stage_durations[uid_wilting])

        # Update plant stage index and remaining stage time.
        self.plants.current_growth_stages[update_condition] = updated_stage_idx_slice
        self.plants.current_remaining_stage_times[self.plants.current_growth_stages < 4] -= 1
        self.plants.current_remaining_stage_times[update_condition] = self.plants.calculate_new_stage_time(
            updated_stage_idx_slice, uid_slices)

        # update health
        self.plants.current_health_status = self.plants.get_health_status()

        # TODO update color

    def _rgetattr(self, obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split('.'))

    def _rsetattr(self, obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self._rgetattr(obj, pre) if pre else obj, post, val)

    def update_current_value_dimension(self, dimension_dict: dict):
        """Sets the parameter values of environment components with those settings.
        Passing a dict with value 'default' string will give the default value
        Passing a dict with value 'random' string will give a purely random value for that dimension
        """

        for dimension, value in dimension_dict.items():
            try:
                garden_prop = self._rgetattr(self, dimension)
            except AttributeError as err:
                print('Skipping dimension as:', err)
                continue
            assert dimension in self.dimensions

            if value == 'default':
                self.dimensions[dimension].current_value = self.dimensions[dimension].default_or_mean_val
            elif value == 'random':  # random
                self.dimensions[dimension].randomize(self.np_random)
            else:
                if type(garden_prop) == np.ndarray:
                    value = np.array(value, dtype=garden_prop.dtype)
                    assert value.size == garden_prop.size
                    self.dimensions[dimension].set(value)
                else:
                    self.dimensions[dimension].set(value)

            self._rsetattr(self, dimension, self.dimensions[dimension].current_value)

    def randomize_all_dimensions(self):
        for dim_name, dim in self.dimensions.items():
            if dim.distribution_type:
                dim.randomize(self.np_random)
                self._rsetattr(self, dim_name, self.dimensions[dim_name].current_value)

    def get_masked_image(self, x_coordinate_array, y_coordinate_array, image, x_0, y_0, y_rows, x_cols,
                         fill_value=0.0):
        """
        Given an array return it with values outside masked window set to fill_value
        Note: This function is able to broadcast with multi dim coordinates and window ranges input arrays
        """
        half_col = (x_cols // 2.)  # + 0.00001
        half_row = (y_rows // 2.)  # + 0.00001
        x_range = np.logical_and(x_coordinate_array >= x_0 - half_col,
                                 x_coordinate_array < x_0 - half_col + x_cols)
        y_range = np.logical_and(y_coordinate_array >= y_0 - half_row,
                                 y_coordinate_array < y_0 - half_row + y_rows)
        result = np.select([np.logical_and(x_range, y_range)], [image], fill_value)
        return result

    def baseline_policy_variable_irrigation(self, observation, time_step=0):
        cc_image = observation['plant_cc_grid']

        water_grid = observation['water_grid']

        water_grid_sectors = self.get_masked_image(self.xv, self.yv, water_grid, self.plants.x_coordinates,
                                                   self.plants.y_coordinates,
                                                   self.sector_rows,
                                                   self.sector_cols)

        health_status_grid = observation['plants_health']

        prune_sector_flag = np.zeros(self.amount_plants, dtype=bool)

        if time_step > self.prune_delay:
            global_cc_vec = self.get_global_cc_soil_vec(cc_image)  # by type + soil

            soil_type_id = max(self.plants.plant_type_ids) + 1
            soil_id = max(self.plants.ids) + 1
            plant_type_and_soil_ids = np.append(self.plants.plant_type_ids, soil_type_id)

            plant_ids_in_prune_windows = self.get_masked_image(self.xv, self.yv, cc_image,
                                                               self.plants.x_coordinates,
                                                               self.plants.y_coordinates,
                                                               self.prune_window_rows,
                                                               self.prune_window_cols,
                                                               fill_value=-1)

            old_ids_per_plant = np.unique(cc_image.ravel())
            cc_values_by_type = plant_type_and_soil_ids[old_ids_per_plant]

            cc_image = self.replace_ids(cc_image, old_ids_per_plant, cc_values_by_type)  # group by plant type

            cc_sum = np.sum(global_cc_vec[0:-1], dtype=np.float32)  # Start from 1 to not include earth in diversity
            if cc_sum != 0:
                prob = global_cc_vec[0:-1] / np.sum(global_cc_vec[0:-1], dtype=np.float32)
            else:
                prob = 0.0

            ratio = 1.36/self.amount_plant_types

            violations = np.where(prob > ratio)[0]

            cc_distributions_in_prune_windows = self.get_masked_image(self.xv, self.yv, cc_image,
                                                                      self.plants.x_coordinates,
                                                                      self.plants.y_coordinates,
                                                                      self.prune_window_rows,
                                                                      self.prune_window_cols,
                                                                      fill_value=-1)

            prune_action_vec = np.zeros(self.amount_plants, dtype=np.int)

            for plant_type_id in violations:
                # determine if sector qualifies to get pruned
                plant_type_area_in_window = np.count_nonzero(cc_distributions_in_prune_windows == plant_type_id,
                                                             axis=(1, 2))
                prune_sector_flag = plant_type_area_in_window > 20

                # prune n largest plants in each prune sector
                plant_ids_windows_to_check = plant_ids_in_prune_windows[prune_sector_flag]
                for i in range(np.count_nonzero(prune_sector_flag)):
                    plant_ids_prune = np.unique(plant_ids_windows_to_check[i])
                    plant_ids_prune = plant_ids_prune[(plant_ids_prune != soil_id) & (plant_ids_prune != -1)]
                    prune_action_vec[plant_ids_prune] = 1  # or += 1 ?

        water_irr_square = self.get_masked_image(self.xv, self.yv, water_grid,
                                                 self.plants.x_coordinates,
                                                 self.plants.y_coordinates,
                                                 self.irr_health_window_width * 2,
                                                 self.irr_health_window_width * 2)
        # print("GRD: ", water_grid_sectors[0][29][29])
        # print("IRR: ", water_irr_square[:][self.plants.x_coordinates, self.plants.y_coordinates])
        # Want the goal to be some arbitrary saturation; let's say 0.2
        # water_ideal = np.ones(water_irr_square)
        # print(self.plants.x_coordinates.shape)
        # Take goal grid - minus current grid = amount to water (current grid is dependent on uptake!)
        # if pos, find scalar value
        # if neg, no watering
        health_irr_square = self.get_masked_image(self.xv, self.yv, health_status_grid,
                                                  self.plants.x_coordinates,
                                                  self.plants.y_coordinates,
                                                  self.irr_health_window_width * 2,
                                                  self.irr_health_window_width * 2)

        ### START CHANGES
        variable_irriation = True
        if variable_irriation:
            x_ind = [i[0][0] for i in self.plants.x_coordinates]
            y_ind = [i[0][0] for i in self.plants.y_coordinates]

            germ_avg = np.mean(self.germination_times)
            matur_avg = np.mean(self.maturation_times) #+ germ_avg
            wait_avg = np.mean(self.waiting_times) + matur_avg
            wilt_avg = np.mean(self.wilting_times) + wait_avg
            # print(germ_avg, matur_avg, wait_avg, wilt_avg)
            const_vwc = 0.2
            if time_step >= germ_avg:
                const_vwc = 0.25
            if time_step >= matur_avg:
                const_vwc = 0.2
            if time_step >= wait_avg:
                const_vwc = 0.1
            if time_step >= wilt_avg:
                const_vwc = 0.0

            curr_water = water_grid[y_ind, x_ind]
            ideal_water = np.ones((self.amount_plants)) * const_vwc 
            diff = ideal_water - curr_water    

            # From calculate_treatment_grid [FROM EXPERIMENTS WITH NOZZLE] 
            """
            - solve for irrigation amount [not used for implementation, but used for real world]
            - solve for [0,?] range of peak gain 
            """
            window_grid_size = np.pi * (self.irr_health_window_width ** 2) / 10000  # in square meters
            k = 1.175  # scaling factor to account for water loss from drainage and etc., determined experimentally
            # Irrigation amount should be the MAX amount we can possibly apply
            peak_gain = (self.irrigation_amount / (window_grid_size * 0.2)) * k  # 1.0
            variable_ratio = diff / peak_gain #how much of max amount (1 == max, 0.5 == max/2)

            # FLAG - Round to discrete amounts
            cont = False
            round_to = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]) #[0.0, .448, .685, 1.0]
            if not cont:
                variable_ratio = [round_to[(np.abs(round_to - i)).argmin()] for i in variable_ratio]

            irr_flag = (diff > np.zeros((self.amount_plants))).astype(np.int) 
            irrigation_flag = irr_flag * variable_ratio
            # print(irr_flag)
            # print("IRR_FLAG: ", irrigation_flag)
            irr_amount = (peak_gain * irrigation_flag) * (window_grid_size * 0.2) / k
            irr_amount *= 1000 #0.2 == 200 mL
            ### END CHANGES
        else:

            post = self.maturation_times + self.waiting_times + self.wilting_times
            post_bool = time_step < post

            total_water_sectors = np.sum(water_grid_sectors, axis=(1, 2))  # water amount per sector
            maximum_water_potential = self.sector_cols * self.sector_rows * self.max_water_content * self.water_threshold
            total_water_sectors += np.sum(water_irr_square, where=health_irr_square == 3, axis=(1, 2))  # overwater contrib
            irrigate_flag = total_water_sectors < maximum_water_potential
            irrigate_flag += np.any(health_irr_square == 1, axis=(1, 2))  # has_underwater(health)
            #Inserted so no watering at death
            irrigate_flag = np.logical_and(irrigate_flag, post_bool)

            irr_amount = irrigate_flag * 0.2

        no_action_vec = np.zeros(self.amount_plants, dtype=np.int)

        return {"irrigation": irrigation_flag, #irrigate_flag.astype(np.int), 
                "prune": prune_sector_flag.astype(np.int),
                "nutrients": no_action_vec,
                "irrigation_volumetric": irr_amount
                }

    def baseline_policy(self, observation, time_step=0):
        cc_image = observation['plant_cc_grid']

        water_grid = observation['water_grid']

        water_grid_sectors = self.get_masked_image(self.xv, self.yv, water_grid, self.plants.x_coordinates,
                                                   self.plants.y_coordinates,
                                                   self.sector_rows,
                                                   self.sector_cols)

        health_status_grid = observation['plants_health']

        prune_sector_flag = np.zeros(self.amount_plants, dtype=bool)

        if time_step > self.prune_delay:
            global_cc_vec = self.get_global_cc_soil_vec(cc_image)  # by type + soil

            soil_type_id = max(self.plants.plant_type_ids) + 1
            soil_id = max(self.plants.ids) + 1
            plant_type_and_soil_ids = np.append(self.plants.plant_type_ids, soil_type_id)

            plant_ids_in_prune_windows = self.get_masked_image(self.xv, self.yv, cc_image,
                                                               self.plants.x_coordinates,
                                                               self.plants.y_coordinates,
                                                               self.prune_window_rows,
                                                               self.prune_window_cols,
                                                               fill_value=-1)

            old_ids_per_plant = np.unique(cc_image.ravel())
            cc_values_by_type = plant_type_and_soil_ids[old_ids_per_plant]

            cc_image = self.replace_ids(cc_image, old_ids_per_plant, cc_values_by_type)  # group by plant type

            cc_sum = np.sum(global_cc_vec[0:-1], dtype=np.float32)  # Start from 1 to not include earth in diversity
            if cc_sum != 0:
                prob = global_cc_vec[0:-1] / np.sum(global_cc_vec[0:-1], dtype=np.float32)
            else:
                prob = 0.0

            ratio = 1.36/self.amount_plant_types

            violations = np.where(prob > ratio)[0]

            cc_distributions_in_prune_windows = self.get_masked_image(self.xv, self.yv, cc_image,
                                                                      self.plants.x_coordinates,
                                                                      self.plants.y_coordinates,
                                                                      self.prune_window_rows,
                                                                      self.prune_window_cols,
                                                                      fill_value=-1)

            prune_action_vec = np.zeros(self.amount_plants, dtype=np.int)

            for plant_type_id in violations:
                # determine if sector qualifies to get pruned
                plant_type_area_in_window = np.count_nonzero(cc_distributions_in_prune_windows == plant_type_id,
                                                             axis=(1, 2))
                prune_sector_flag = plant_type_area_in_window > 20

                # prune n largest plants in each prune sector
                plant_ids_windows_to_check = plant_ids_in_prune_windows[prune_sector_flag]
                for i in range(np.count_nonzero(prune_sector_flag)):
                    plant_ids_prune = np.unique(plant_ids_windows_to_check[i])
                    plant_ids_prune = plant_ids_prune[(plant_ids_prune != soil_id) & (plant_ids_prune != -1)]
                    prune_action_vec[plant_ids_prune] = 1  # or += 1 ?

        water_irr_square = self.get_masked_image(self.xv, self.yv, water_grid,
                                                 self.plants.x_coordinates,
                                                 self.plants.y_coordinates,
                                                 self.irr_health_window_width * 2,
                                                 self.irr_health_window_width * 2)

        health_irr_square = self.get_masked_image(self.xv, self.yv, health_status_grid,
                                                  self.plants.x_coordinates,
                                                  self.plants.y_coordinates,
                                                  self.irr_health_window_width * 2,
                                                  self.irr_health_window_width * 2)

        total_water_sectors = np.sum(water_grid_sectors, axis=(1, 2))  # water amount per sector
        maximum_water_potential = self.sector_cols * self.sector_rows * self.max_water_content * self.water_threshold
        total_water_sectors += np.sum(water_irr_square, where=health_irr_square == 3, axis=(1, 2))  # overwater contrib
        irrigate_flag = total_water_sectors < maximum_water_potential

        irrigate_flag += np.any(health_irr_square == 1, axis=(1, 2))  # has_underwater(health)

        no_action_vec = np.zeros(self.amount_plants, dtype=np.int)

        irr_amount = irrigate_flag * 0.2
        # print(irrigate_flag)
        # print(irr_amount)

        return {"irrigation": irrigate_flag.astype(np.int),
                "prune": prune_sector_flag.astype(np.int),
                "nutrients": no_action_vec,
                "irrigation_volumetric": irr_amount
                }

    """def get_observation(self):
        water_grid = self.soil.water_content_grid
        plants_unoccluded_ratio = self.plants.unoccluded_ratio
        plants_health = self.plants.current_health_status
        # plants_structure = self.plants.current_structure / self.plants.reference_structure
        plants_structure = self.plants.current_outer_radii / self.plants.reference_outer_radii
        return {'water_grid': water_grid,
                'plants_unoccluded_ratios': plants_unoccluded_ratio,
                'plants_health': plants_health,
                'plants_structure': plants_structure}"""

    def get_observation(self):
        cc_image = self.get_canopy_image()
        water_grid = self.soil.water_content_grid
        # nuts_grid = TODO
        health_cc_image = np.zeros((self.rows, self.columns), dtype=np.int)
        visible_ids = np.unique(np.ravel(cc_image))
        for plant_id in visible_ids[:-1]:
            health_cc_image[cc_image == plant_id] = self.plants.current_health_status[plant_id]
        return {'plant_cc_grid': cc_image,
                'water_grid': water_grid,
                'plants_health': health_cc_image,
                # TODO nuts
                }

    def get_canopy_image(self):
        plant_height_index = self.plants.get_sorted_descending_heights_indices()
        plant_soil_height_index = np.append(plant_height_index, len(plant_height_index))

        # stacked grids with plant and soil location sorted by height
        sorted_plants_soil_grids = np.append(self.plants.current_grid_locations, np.ones(
            (1, self.rows, self.columns)), axis=0)[plant_soil_height_index]

        cc_image = sorted_plants_soil_grids.argmax(axis=0)  # generate top down canopy coverage image
        values = np.unique(cc_image.ravel())
        true_visible_ids = plant_soil_height_index[values]  # get true ids
        cc_image = self.replace_ids(cc_image, values, true_visible_ids)  # replace

        return cc_image  # with with visible plant ids

    def get_soil_sensor_readings(self, water_grid):
        window = 2 * self.irr_health_window_width
        sensor_sectors = self.get_masked_image(self.xv, self.yv, water_grid,
                                               self.plants.x_coordinates,
                                               self.plants.y_coordinates,
                                               window,
                                               window)

        return np.sum(sensor_sectors, axis=(1, 2)) / (window * window)


    def get_global_cc_soil_vec(self, cc_image) -> np.ndarray:
        soil_plants_identifier, soil_plants_cc_counts = np.unique(cc_image, return_counts=True)

        cc_per_plant_soil = np.zeros(1 + self.amount_plants)  # temp vector
        cc_per_plant_soil[soil_plants_identifier] = soil_plants_cc_counts  # map counts by sorted ids

        plant_type_and_soil_ids = np.append(self.plants.plant_type_ids,
                                            max(self.plants.plant_type_ids) + 1)  # app. soil(type) id

        return np.bincount(plant_type_and_soil_ids, weights=cc_per_plant_soil)  # sum cc counts by plant type

    def get_metrics(self, cc_image):
        cc_per_plant_type_soil = self.get_global_cc_soil_vec(cc_image)
        cc_per_plant_type = cc_per_plant_type_soil[:-1]  # without soil
        total_cc = np.sum(cc_per_plant_type)
        coverage = total_cc / (self.rows * self.columns)
        prob = cc_per_plant_type[np.nonzero(cc_per_plant_type)] / total_cc
        entropy = np.sum(-prob * np.log(prob))

        if self.amount_plant_types > 1:
            diversity = entropy / np.log(self.amount_plant_types)  # normalized entropy
        else:
            diversity = 1.0

        return coverage, diversity

    def calculate_multi_model_entropy(self, cc_image):
        """ Calculate multi-model entropy (diversity and total plant coverage)"""

        mme_1_global_cc_vec = self.get_global_cc_soil_vec(cc_image)

        mme_1_global_prob = (mme_1_global_cc_vec[np.nonzero(mme_1_global_cc_vec)]) / (self.rows * self.columns)
        mme_1_global_entropy = np.sum(-mme_1_global_prob * np.log(mme_1_global_prob))
        return mme_1_global_entropy / np.log(1 + self.plants.plant_type_ids.size)  # normalized entropy

    def step(self, action: dict):
        done = False
        self.current_day += 1

        # Retrieve action from input and supplement missing actions with non interfering placeholders
        no_action_vec = np.zeros(self.amount_plants, dtype=np.int)
        irrigation_actions = action.get("irrigation", no_action_vec)
        nutrient_actions = action.get("nutrients", no_action_vec)
        prune_actions = action.get("prune", no_action_vec)
        irrigation_volumetric = action.get("irrigation_volumetric", no_action_vec)

        # convert irrigation action to a matching action in simulator grid form
        water_action_grid = self.soil.calculate_treatment_grid(x=self.xv, y=self.yv,
                                                               treatment_actions=irrigation_actions.reshape(
                                                                   self.amount_plants, 1, 1),
                                                               x_0=self.plants.x_coordinates,
                                                               y_0=self.plants.y_coordinates,
                                                               r_0=4, gain_slope=1 / 128,
                                                               amount=self.irrigation_amount)  # gain_slope=15

        #  convert nuts action to a matching action in simulator grid form
        nutrient_action_grid = self.soil.calculate_treatment_grid(x=self.xv, y=self.yv,
                                                                  treatment_actions=nutrient_actions.reshape(
                                                                      self.amount_plants, 1, 1),
                                                                  x_0=self.plants.x_coordinates,
                                                                  y_0=self.plants.y_coordinates,
                                                                  r_0=4, gain_slope=15,
                                                                  amount=self.irrigation_amount)  # TODO add function

        # prune_actions = np.zeros(self.amount_plants)  # Manually disable pruning

        self.simulation_step(water_action_grid, nutrient_action_grid, prune_actions)

        cc_image = self.get_canopy_image()

        # calculate normalized irrigation total,  1 liter is estimated to be the maximum per plant/action
        self.irrigation = np.sum(irrigation_actions * self.irrigation_amount) / (self.plants.amount_plants * 0.001)
        #RECALCULATE for variable irrigation
        # self.total_irrigation = np.sum(irrigation_actions * self.irrigation_amount) 
        self.total_irrigation = np.sum(irrigation_volumetric) 


        coverage, diversity = self.get_metrics(cc_image)
        self.coverages.append(coverage)
        self.diversities.append(diversity)
        self.irrigation_amounts.append(self.total_irrigation)

        reward = self.calculate_multi_model_entropy(cc_image)

        obs = self.get_observation()

        # For growth
        # self.save_radii()

        if self.current_day == self.day_limit:
            done = True

        info = {'overwater_deaths': self.overwater_deaths,
                'underwater_deaths': self.underwater_deaths,
                'coverage': coverage,
                'diversity': diversity,
                'num_irr': np.sum(irrigation_actions),
                'num_prune': np.sum(prune_actions),
                'total_irr': self.total_irrigation,
                'irr_amounts':irrigation_volumetric,
                # TODO nuts_actions
                }

        return obs, reward, done, info

    def reset(self):
        # Todo save last metrics?
        self.randomize_all_dimensions()
        self.current_day = 0
        self.weather.reset()
        self.soil.reset(self.np_random)
        self.plants = Plants(rows=self.rows, columns=self.columns, xv=self.xv, yv=self.yv,
                             amount_plants=self.amount_plants, amount_plant_types=self.amount_plant_types,
                             overwatered_time_threshold=self.overwatered_time_threshold,
                             underwatered_time_threshold=self.underwatered_time_threshold,
                             overwater_threshold=self.overwatered_threshold,
                             underwater_threshold=self.underwaterd_threshold,
                             reference_outer_radii=self.reference_outer_radii, common_names=self.common_names,
                             germination_times=self.germination_times,
                             maturation_times=self.maturation_times,
                             waiting_stage_durations=self.waiting_times,
                             wilting_stage_durations=self.wilting_times,
                             water_use_efficiencies=self.water_use_efficiencies,
                             light_use_efficiencies=self.light_use_efficiencies,
                             nutrients_use_efficiencies=self.nutrients_use_efficiencies,
                             x_coordinates=self.x_coordinates, y_coordinates=self.y_coordinates)

        self.water_uptake_queue = [None for _ in range(self.tau)]
        # [[None for _ in range(self.tau[i])] for i in range(AMOUNT_PLANTS)]
        self.nutrients_uptake_queue = [None for _ in range(self.tau)]
        self.available_water_queue = [None for _ in range(self.tau)]
        self.desired_water_queue = [None for _ in range(self.tau)]
        self.desired_nutrients_queue = [None for _ in range(self.tau)]

        self.current_irrigation_total = 0.0
        self.current_fertilizer_total = 0.0
        self.coverages = []
        self.diversities = []
        self.irrigation_amounts = []

        obs = self.get_observation()
        return obs

    def replace_ids(self, cc_image, old_ids, new_ids):
        n_min, n_max = cc_image.min(), cc_image.max()
        replacer = np.arange(n_min, n_max + 1)
        # Mask replacements out of range
        mask = (old_ids >= n_min) & (old_ids <= n_max)
        replacer[old_ids[mask] - n_min] = new_ids[mask]
        return replacer[cc_image - n_min]

    def render(self, mode='human', close=False, sleep=True):
        assert mode in ["human", "save"], "Invalid mode, must be either \"human\", \"save\""
        if mode == "human":

            data = self.get_observation()

            cc_image = data['plant_cc_grid']
            water = data['water_grid']
            health = data['plants_health']

            plant_type_and_soil_ids = np.append(self.plants.plant_type_ids,
                                                max(self.plants.plant_type_ids) + 1)

            old_ids_per_plant = np.unique(cc_image.ravel())
            cc_values_by_type = plant_type_and_soil_ids[old_ids_per_plant]

            cc_image = self.replace_ids(cc_image, old_ids_per_plant, cc_values_by_type)  # cc plant type not plant id!
            # plant_type_soil_ids = range(self.amount_plant_types + 1)
            # unique_cn, unique_cn_index = np.unique(self.plants.common_names, return_index=True)
            unique_types, unique_index = np.unique(plant_type_and_soil_ids, return_index=True)

            fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(8, 3.5))
            ax = axs[0]
            ax.set_title('CC ' + 'Day ' + str(self.current_day), pad=12)
            im = ax.imshow(cc_image, cmap='jet', interpolation='none', vmin=0, vmax=self.amount_plant_types + 1)
            colors = [im.cmap(im.norm(i)) for i in unique_types]

            # create a patch (proxy artist) for every color
            patches = [mpatches.Patch(color=colors[i],
                                      label="{cc_type}".format(
                                          cc_type=self.plants.common_names[unique_index[i]] if i != max(
                                              unique_types) else "Soil")
                                      ) for i in unique_types]
            # put those patched as legend-handles into the legend
            ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2,
                      fontsize='x-small')

            ax = axs[1]
            ax.set_title('Water grid', pad=12)
            pcm = ax.imshow(water, cmap='Blues')
            fig.colorbar(pcm, ax=ax, location='bottom')

            ax = axs[2]
            ax.set_title('Health gird', pad=12)
            cmap = plt.get_cmap('RdBu', np.max(health).astype(np.int) - np.min(health).astype(np.int) + 1)
            pcm = ax.matshow(health, cmap=cmap, vmin=np.min(health) - .5, vmax=np.max(health) + .5)
            fig.colorbar(pcm, ax=ax, ticks=np.arange(np.min(health), np.max(health) + 1), location='bottom')

            for ax in axs:
                ax.set_anchor('N')
                ax.set_xticks(np.linspace(0, self.columns-1, num=7, endpoint=True, dtype=int))
                ax.set_yticks(np.linspace(0, self.rows-1, num=7, endpoint=True, dtype=int))
                ax.tick_params(axis="both", bottom=True, right=True, top=True, labelbottom=False, labeltop=True)

            plt.show()
            if sleep:
                time.sleep(0.8)
            if close:
                plt.close()

        if mode == "save":
            cc_image = self.get_canopy_image()
            cmap = plt.cm.jet
            norm = plt.Normalize(vmin=0, vmax=np.max(self.plants.ids))
            image = cmap(norm(cc_image))
            plt.imsave("./data/canopy/canopy_" + str(self.current_day) + ".png", image)

    def save_water_grid(self):
        os.makedirs('./data/water_grid/', exist_ok=True)
        water_grid = self.get_observation()['water_grid']
        grid = water_grid / 0.3 * 255  # normalized by max water threshold which is 0.3
        array = np.zeros((self.rows, self.columns, 3), dtype=np.uint8)
        array[:, :, 2] = grid
        img = Image.fromarray(array.astype(np.uint8), 'RGB')
        img2 = img.resize((600, 600), Image.ANTIALIAS)
        img2 = ImageOps.expand(img2, border=300, fill='white')
        img2.save("./data/water_grid/grid_" + str(self.current_day) + ".png")


class ContPruneIrrActionNormObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prune_rate = 1.0  # reset to pass prune action w/o rate
        self.irrigation_amount = 0.001  # 1 liter in cubic meter

        self.observation_space = spaces.Dict(
            {"norm_cc_vec": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plant_types + 1,)),
             "norm_water_sensor_vec": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants, ), dtype=np.float32),
             })

        self.action_space = spaces.Dict({
            "prune": spaces.Box(low=-1.0, high=1.0, shape=(self.amount_plants,)),
            "irrigation": spaces.Box(low=-1.0, high=1.0, shape=(self.amount_plants,))
        })

    # Todo: implement reward function to account for irrigation costs
    def reward(self, reward):
        return reward - self.env.irrigation

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, act):
        observation, reward, done, info = self.env.step(self.action(act))
        return self.observation(observation), self.reward(reward), done, info

    def action(self, act):
        return {"irrigation": (act['irrigation'] + 1) / 2,  # irrigation_amount * (action in [0, 1])
                "nutrients": np.zeros(self.amount_plants, dtype=np.int),
                "prune": (act['prune'] + 1) / 5  # prune_rate * (action in [0, 0.4]
                }

    def observation(self, obs):
        cc_image = obs['plant_cc_grid']
        water_grid = obs['water_grid']
        normalized_cc_vec = (self.get_global_cc_soil_vec(cc_image) / (self.rows * self.columns)).astype(np.float32)
        normalized_water_grid = water_grid / self.water_saturation_thr
        norm_water_sensor_vec = self.get_soil_sensor_readings(normalized_water_grid).astype(np.float32)

        return {'norm_cc_vec': normalized_cc_vec,
                'norm_water_sensor_vec': norm_water_sensor_vec,
                }


class DisPruneIrrActionNormObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prune_rate = 1.0
        self.prune_action_map = np.array([0.0, 0.07, 0.14, 0.21, 0.28])

        self.observation_space = spaces.Dict(
            {"norm_cc_vec": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plant_types + 1,)),
             "norm_water_sensor_vec": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants, ), dtype=np.float32),
             })

        self.action_space = spaces.Dict({
            "prune": spaces.MultiDiscrete([self.prune_action_map.size for _ in range(self.amount_plants)]),
            "irrigation": spaces.MultiDiscrete([2 for _ in range(self.amount_plants)])
            #"irrigation": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants,), dtype=np.int)
        })

    # Todo: implement reward function to account for irrigation costs
    def reward(self, reward):
        return np.maximum(reward - 0.1 * self.env.irrigation, 0.0)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, act):
        observation, reward, done, info = self.env.step(self.action(act))
        return self.observation(observation), self.reward(reward), done, info

    def action(self, act):
        return {"irrigation": act['irrigation'],  # irrigation_amount * (action in [0, 1])
                "nutrients": np.zeros(self.amount_plants, dtype=np.int),
                "prune": self.prune_action_map[act['prune']]  # map prune rate * (action in [0, 0.4]
                }

    def observation(self, obs):
        cc_image = obs['plant_cc_grid']
        water_grid = obs['water_grid']
        normalized_cc_vec = (self.get_global_cc_soil_vec(cc_image) / (self.rows * self.columns)).astype(np.float32)
        normalized_water_grid = water_grid / self.water_saturation_thr
        norm_water_sensor_vec = self.get_soil_sensor_readings(normalized_water_grid).astype(np.float32)

        return {'norm_cc_vec': normalized_cc_vec,
                'norm_water_sensor_vec': norm_water_sensor_vec,
                }


class BinPruneIrrActionNormObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict(
            {"norm_cc_vec": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plant_types + 1,)),
             "norm_water_sensor_vec": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants, ), dtype=np.float32),
             })

        self.action_space = spaces.Dict({
            "prune": spaces.MultiDiscrete([2 for _ in range(self.amount_plants)]),
            # "prune": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants,), dtype=np.int),
            "irrigation": spaces.MultiDiscrete([2 for _ in range(self.amount_plants)])
            # "irrigation": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants,), dtype=np.int)
        })

    # Todo: implement reward function to account for irrigation costs
    def reward(self, reward):
        return np.maximum(reward - 0.1 * self.env.irrigation, 0.0)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, act):
        observation, reward, done, info = self.env.step(self.action(act))
        return self.observation(observation), self.reward(reward), done, info

    def action(self, act):
        return {"irrigation": act['irrigation'],  # irrigation_amount * (action in {0, 1})
                "nutrients": np.zeros(self.amount_plants, dtype=np.int),
                "prune": act['prune']  # prune_rate * (action in {0, 0.4})
                }

    def observation(self, obs):
        cc_image = obs['plant_cc_grid']
        water_grid = obs['water_grid']
        normalized_cc_vec = (self.get_global_cc_soil_vec(cc_image) / (self.rows * self.columns)).astype(np.float32)
        normalized_water_grid = water_grid / self.water_saturation_thr
        norm_water_sensor_vec = self.get_soil_sensor_readings(normalized_water_grid).astype(np.float32)

        return {'norm_cc_vec': normalized_cc_vec,
                'norm_water_sensor_vec': norm_water_sensor_vec,
                }


class ContPruneBaseActNormObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prune_rate = 1.0  # reset to pass prune action w/o rate

        self.observation_space = spaces.Dict(
            {"norm_cc_vec": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plant_types + 1,)),
             })

        self.action_space = spaces.Dict({
            "prune": spaces.Box(low=-1.0, high=1.0, shape=(self.amount_plants,))
        })

    def observation(self, obs):
        cc_image = obs['plant_cc_grid']
        normalized_cc_vec = (self.get_global_cc_soil_vec(cc_image) / (self.rows * self.columns)).astype(np.float32)
        return {'norm_cc_vec': normalized_cc_vec,
                }

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, act):
        observation, reward, done, info = self.env.step(self.action(act))
        return self.observation(observation), reward, done, info

    def action(self, act):
        obs = self.get_observation()
        baseline_action = self.baseline_policy(observation=obs, time_step=self.current_day)

        return {"irrigation": baseline_action['irrigation'],
                "nutrients": baseline_action['nutrients'],
                "prune": (act['prune'] + 1) / 5  # prune range [0, 0.4]
                }


class DisPruneBaseActNormObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prune_rate = 1.0
        self.prune_action_map = np.array([0.0, 0.07, 0.14, 0.21, 0.28])

        self.observation_space = spaces.Dict(
            {"norm_cc_vec": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plant_types + 1,)),
             })

        self.action_space = spaces.Dict({
            "prune": spaces.MultiDiscrete([self.prune_action_map.size for _ in range(self.amount_plants)]),
        })

    def observation(self, obs):
        cc_image = obs['plant_cc_grid']
        normalized_cc_vec = (self.get_global_cc_soil_vec(cc_image) / (self.rows * self.columns)).astype(np.float32)
        return {'norm_cc_vec': normalized_cc_vec,
                }

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, act):
        observation, reward, done, info = self.env.step(self.action(act))
        return self.observation(observation), reward, done, info

    def action(self, act):
        obs = self.get_observation()
        baseline_action = self.baseline_policy(observation=obs, time_step=self.current_day)

        return {"irrigation": baseline_action['irrigation'],
                "nutrients": np.zeros(self.amount_plants, dtype=np.int),
                "prune": self.prune_action_map[act['prune']]  # map prune rates in {0.0, 0.05, 0.1, 0.16, 0.2, 0.3, 0.4}
                }


class BinPruneBaseActNormObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict(
            {"norm_cc_vec": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plant_types + 1,)),
             })

        self.action_space = spaces.Dict({
            "prune": spaces.MultiDiscrete([2 for _ in range(self.amount_plants)]),
            # "prune": spaces.Box(low=0.0, high=1.0, shape=(self.amount_plants,), dtype=np.int),
        })

    def observation(self, obs):
        cc_image = obs['plant_cc_grid']
        normalized_cc_vec = (self.get_global_cc_soil_vec(cc_image) / (self.rows * self.columns)).astype(np.float32)
        return {'norm_cc_vec': normalized_cc_vec,
                }

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, act):
        observation, reward, done, info = self.env.step(self.action(act))
        return self.observation(observation), reward, done, info

    def action(self, act):
        obs = self.get_observation()
        baseline_action = self.baseline_policy(observation=obs, time_step=self.current_day)

        return {"irrigation": baseline_action['irrigation'],
                "nutrients": baseline_action['nutrients'],
                "prune": act['prune']  # prune action in {0, 1} * prune_rate 0.15
                }
