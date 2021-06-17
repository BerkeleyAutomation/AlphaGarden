import gym
import torch
from simulator.sim_globals import NUM_IRR_ACTIONS, NUM_PLANTS, PERCENT_NON_PLANT_CENTERS, PRUNE_DELAY, ROWS, COLS, SECTOR_ROWS, SECTOR_COLS, PRUNE_WINDOW_ROWS, PRUNE_WINDOW_COLS, STEP, SOIL_MOISTURE_SENSOR_ACTIVE, SOIL_MOISTURE_SENSOR_POSITIONS, GARDEN_START_DATE
import simalphagarden
from net import Net
from constants import TrainingConstants
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import multiprocessing as mp
import time
import json
import math
from urllib.request import urlopen
from datetime import datetime, timezone
import os
from time import gmtime, strftime


def getDeviceReadingsMostRecent():
    buffer = 1800 # 1/2 hour buffer for querying the logger again
    user = "mpresten@berkeley.edu"
    user_password = "AlphaGard3n"
    device_serial_number = "z6-08807"
    device_password = "24453-30317"
    ip = "zentracloud.com"
    utc_time = datetime.now(timezone.utc)

    create = False
    utc_timestamp = utc_time.timestamp()    
    most_recent = round(utc_timestamp)
    if (os.path.exists("soil_moisture.txt")):
        f = open("soil_moisture.txt", "r")
        t = f.readline()
        s = f.readline()
        slist = f.readline()
        f.close()
    else:
        create = True
    if (create or ((int(t) + buffer) < most_recent)) :
        first = True
        count = 0
        while (first or json.dumps(readings_json['device']['timeseries']) == "[]"):
            response = urlopen('https://' + ip + '/api/v1/readings'
                                + '?' + "user=" + user
                                + '&' + "user_password=" + user_password
                                + '&' + "sn=" + device_serial_number
                                + '&' + "device_password=" + device_password
                                + '&' + "start_time=" + str(most_recent - count)
                                )
            readings_str = response.read()
            readings_json = json.loads(readings_str)
            first = False
            count += 1800
        # Readings are now contained in the 'readings_json' Python dictionary

        print("once")
        # Examples of accessing data
        time = int(float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][-1][0])))
        s1 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][-1][3][0]['value']))
        s2 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][-1][4][0]['value']))
        s3 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][-1][5][0]['value']))
        s4 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][-1][6][0]['value']))
        s5 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][-1][7][0]['value']))
        s6 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][-1][8][0]['value']))
        slist = [s1, s2, s3, s4, s5, s6]

        if (os.path.exists("soil_moisture.txt")):
            os.remove("soil_moisture.txt")

        f = open("soil_moisture.txt", "w")
        f.write(str(most_recent)) # time when the logger was pinged not when the measurements were taken
        f.write("\nSoil Readings: \n" + str(slist))
        f.write("\nThe Actual Time the Readings were Taken: " + str(time))
        f.close()
    else:
        slist = json.loads(slist)
        s1 = float(json.dumps(slist[0]))
        s2 = float(json.dumps(slist[1]))
        s3 = float(json.dumps(slist[2]))
        s4 = float(json.dumps(slist[3]))
        s5 = float(json.dumps(slist[4]))
        s6 = float(json.dumps(slist[5]))

    return s1, s2, s3, s4, s5, s6

def getDeviceReadings(current_day, adjust=0):
    """ Gets the soil moisture readings.
        Args:
            current_day (int): The day to get the readings.
            adjust (int): Gives readings from the current day with an adjustment of ADJUST seconds
    """
    user = "mpresten@berkeley.edu"
    user_password = "AlphaGard3n"
    device_serial_number = "z6-08807"
    device_password = "24453-30317"
    ip = "zentracloud.com"
    create = False
    utc_time = datetime.now(timezone.utc)
    utc_timestamp = utc_time.timestamp()
    current_time = round(utc_timestamp)
    most_recent = current_day * 86400 + GARDEN_START_DATE + adjust
    slist = None
    if (most_recent > current_time):
        return None, None, None, None, None, None

    if (os.path.exists("policy_metrics/soil_moisture.txt")):
        f = open("policy_metrics/soil_moisture.txt", "r")
        slist = f.readline()
        f.close()
        slist = json.loads(slist)   

    if (slist == None or str(most_recent) not in slist):
        temp = {}
        response = urlopen('https://' + ip + '/api/v1/readings'
                            + '?' + "user=" + user
                            + '&' + "user_password=" + user_password
                            + '&' + "sn=" + device_serial_number
                            + '&' + "device_password=" + device_password
                            + '&' + "start_time=" + str(most_recent)
                            )
        readings_str = response.read()
        readings_json = json.loads(readings_str)
        # Readings are now contained in the 'readings_json' Python dictionary
        # Examples of accessing data
        time = int(float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][0][0])))
        print(time)
        s1 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][0][3][0]['value']))
        s2 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][0][4][0]['value']))
        s3 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][0][5][0]['value']))
        s4 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][0][6][0]['value']))
        s5 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][0][7][0]['value']))
        s6 = float(json.dumps(readings_json['device']['timeseries'][0]['configuration']['values'][0][8][0]['value']))
        print(most_recent, time)
        temp[most_recent] = [s1, s2, s3, s4, s5, s6]
        if (os.path.exists("policy_metrics/soil_moisture.txt")):
            os.remove("policy_metrics/soil_moisture.txt")
        f = open("policy_metrics/soil_moisture.txt", "w")

        if slist == None:
            json.dump(temp, f)
        else:
            slist.update(temp)
            json.dump(slist, f)

        f.close()

    else:
        s1 = float(json.dumps(slist[str(most_recent)][0]))
        s2 = float(json.dumps(slist[str(most_recent)][1]))
        s3 = float(json.dumps(slist[str(most_recent)][2]))
        s4 = float(json.dumps(slist[str(most_recent)][3]))
        s5 = float(json.dumps(slist[str(most_recent)][4]))
        s6 = float(json.dumps(slist[str(most_recent)][5]))

    return s1, s2, s3, s4, s5, s6

def save_water_grid(water_grid, current_day, purpose=""):
    np.savetxt("policy_metrics/water_grid/water_grid_" + str(current_day) + purpose + ".txt", np.array(water_grid), fmt="%f")

    with open('policy_metrics/water_grid/water_grid_' + str(current_day) + purpose + '.pkl', 'wb') as f:
        pickle.dump(water_grid, f)

    return

def averagedReadings(water_grid, s_pos, avg=True):
    if avg:
        readings = []
        water_grid[s_pos[0]][s_pos[1]]
        h, k = s_pos[0] + 1, s_pos[1] + 2
        a, b = 3, 4
        for x in range(s_pos[0] - 2, s_pos[0] + 5):
            for y in range(s_pos[1] - 2, s_pos[1] + 7):
                x, y = round(x), round(y)
                p = ((math.pow((x - h), 2) // math.pow(a, 2)) + (math.pow((y - k), 2) // math.pow(b, 2))) 
                if (p <= 1 and x >=0 and y>= 0 and x < ROWS and y < COLS):
                    readings.append(water_grid[x][y])
        return sum(readings)/len(readings)
    else:
        return water_grid[s_pos[0]][s_pos[1]]

def augment_soil_moisture_map(water_grid, trial, current_day, max_days, error_dict):
    start_buffer = 0 # number of days after the start of the garden that the sensors are sending accurate moisture levels
    end_buffer = 0 # number of days before the end of the garden that the sensors are sending accurate moisture levels

    s1_avg, s2_avg, s3_avg, s4_avg, s5_avg, s6_avg, s1_pavg, s2_pavg, s3_pavg, s4_pavg, s5_pavg, s6_pavg = error_dict["s1_avg"], \
        error_dict["s2_avg"], error_dict["s3_avg"], error_dict["s4_avg"], error_dict["s5_avg"], error_dict["s6_avg"], \
        error_dict["s1_pavg"], error_dict["s2_pavg"], error_dict["s3_pavg"], error_dict["s4_pavg"], error_dict["s5_pavg"], error_dict["s6_pavg"]
    s1, s2, s3, s4, s5, s6 = getDeviceReadings(current_day-1)

    s1_pos = SOIL_MOISTURE_SENSOR_POSITIONS[0]
    s2_pos = SOIL_MOISTURE_SENSOR_POSITIONS[1]
    s3_pos = SOIL_MOISTURE_SENSOR_POSITIONS[2]
    s4_pos = SOIL_MOISTURE_SENSOR_POSITIONS[3]
    s5_pos = SOIL_MOISTURE_SENSOR_POSITIONS[4]
    s6_pos = SOIL_MOISTURE_SENSOR_POSITIONS[5]

    if not os.path.exists('policy_metrics/water_grid'):
        os.makedirs('policy_metrics/water_grid')

    s1_error, s2_error, s3_error, s4_error, s5_error, s6_error = None, None, None, None, None, None
    s1_perror, s2_perror, s3_perror, s4_perror, s5_perror, s6_perror = None, None, None, None, None, None
    s1_sim, s2_sim, s3_sim, s4_sim, s5_sim, s6_sim = None, None, None, None, None, None
    #calculating absolute and percent error
    #percent error = |(actual - expected)/expected| where expected = sensor measurement, actual = val in sim
    f = open("policy_metrics/water_grid_error_ " + str(trial) + ".txt", "a")
    if s1 == None or current_day < start_buffer or current_day > max_days - end_buffer:
        f.write("\nDay " + str(current_day) + ":")
        f.write("\n\tActual Reading: ")
        f.write("\n\t\tS1: None S2: None S3: None S4: None S5: None S6: None")
        f.write("\n\tValue in Sim: ")
        f.write("\n\t\tS1: " + str(s1_sim) + " S2: " + str(s2_sim) + " S3: "  + str(s3_sim) + " S4: " + str(s4_sim) + " S5: " + str(s5_sim) + " S6: " + str(s6_sim))
        f.write("\n\tAbsolute Error: ")
        f.write("\n\t\tS1 Error: " + str(s1_error) + " S2 Error: " + str(s2_error) + " S3 Error: "  + str(s3_error) + " S4 Error: " + str(s4_error) + " S5 Error: " + str(s5_error) + " S6 Error: " + str(s6_error))
        f.write("\n\tPercent Error: ")
        f.write("\n\t\tS1 Error: " + str(s1_perror) + " S2 Error: " + str(s2_perror) + " S3 Error: "  + str(s3_perror) + " S4 Error: " + str(s4_perror) + " S5 Error: " + str(s5_perror) + " S6 Error: " + str(s6_perror))
        save_water_grid(water_grid, current_day, "right_after_watering")
        if current_day == max_days:
            s1_a, s1_pa = None, None
            s2_a, s2_pa = None, None
            s3_a, s3_pa = None, None
            s4_a, s4_pa = None, None
            s5_a, s5_pa = None, None
            s6_a, s6_pa = None, None
            if len(s1_avg) != 0:
                s1_a, s1_pa  = sum(s1_avg)/len(s1_avg), sum(s1_pavg)/len(s1_pavg)
            if len(s2_avg) != 0:
                s2_a, s2_pa  = sum(s2_avg)/len(s2_avg), sum(s2_pavg)/len(s2_pavg)
            if len(s3_avg) != 0:
                s3_a, s3_pa  = sum(s3_avg)/len(s3_avg), sum(s3_pavg)/len(s3_pavg)
            if len(s4_avg) != 0:
                s4_a, s4_pa  = sum(s4_avg)/len(s4_avg), sum(s4_pavg)/len(s4_pavg)
            if len(s5_avg) != 0:
                s5_a, s5_pa  = sum(s5_avg)/len(s5_avg), sum(s5_pavg)/len(s5_pavg)
            if len(s6_avg) != 0:
                s6_a, s6_pa  = sum(s6_avg)/len(s6_avg), sum(s6_pavg)/len(s6_pavg)
            f.write("\n\nAverage Absolute Error: ")
            f.write("\n\t\tS1: " + str(s1_a) + " S2: " + str(s2_a) + " S3: "  + str(s3_a) + " S4: " + str(s4_a) + " S5: " + str(s5_a) + " S6: " + str(s6_a))
            f.write("\nAverage Percent Error: ")
            f.write("\n\t\tS1: " + str(s1_pa) + " S2: " + str(s2_pa) + " S3: "  + str(s3_pa) + " S4: " + str(s4_pa) + " S5: " + str(s5_pa) + " S6: " + str(s6_pa))        
        return None

    m = max(s1, s2, s3, s4, s5, s6)
    if SOIL_MOISTURE_SENSOR_ACTIVE[0]:
        s1_sim = averagedReadings(water_grid, s1_pos)
        s1_error = abs(s1_sim - s1)
        s1_perror = abs((s1_sim - s1)/s1) * 100
        s1_avg.append(s1_error)
        s1_pavg.append(s1_perror)
        #env.wrapper_env.set_water_grid(s1, s1_pos)
    else:
        s1 = None

    if SOIL_MOISTURE_SENSOR_ACTIVE[1]:
        s2_sim = averagedReadings(water_grid, s2_pos)
        s2_error = abs(s2_sim - s2)
        s2_perror = abs((s2_sim - s2)/s2) * 100
        s2_avg.append(s2_error)
        s2_pavg.append(s2_perror)
        #env.wrapper_env.set_water_grid(s2, s2_pos)
    else:
        s2 = None
    if SOIL_MOISTURE_SENSOR_ACTIVE[2]:
        s3_sim = averagedReadings(water_grid, s3_pos)
        s3_error = abs(s3_sim - s3)
        s3_perror = abs((s3_sim - s3)/s3) * 100
        s3_avg.append(s3_error)
        s3_pavg.append(s3_perror)
        #env.wrapper_env.set_water_grid(s3, s3_pos)
    else:
        s3 = None
    if SOIL_MOISTURE_SENSOR_ACTIVE[3]:
        s4_sim = averagedReadings(water_grid, s4_pos)
        s4_error = abs(s4_sim - s4)
        s4_perror = abs((s4_sim - s4)/s4) * 100
        s4_avg.append(s4_error)
        s4_pavg.append(s4_perror)
        #env.wrapper_env.set_water_grid(s4, s4_pos)
    else:
        s4 = None
    if SOIL_MOISTURE_SENSOR_ACTIVE[4]:
        s5_sim = averagedReadings(water_grid, s5_pos)
        s5_error = abs(s5_sim - s5)
        s5_perror = abs((s5_sim - s5)/s5) * 100
        s5_avg.append(s5_error)
        s5_pavg.append(s5_perror)
        #env.wrapper_env.set_water_grid(s5, s5_pos)
    else:
        s5 = None
    if SOIL_MOISTURE_SENSOR_ACTIVE[5]:
        s6_sim = averagedReadings(water_grid, s6_pos)
        s6_error = abs(s6_sim - s6)
        s6_perror = abs((s6_sim - s6)/s6) * 100
        s6_avg.append(s6_error)
        s6_pavg.append(s6_perror)
        #env.wrapper_env.set_water_grid(s6, s6_pos)
    else:
        s6 = None

    f.write("\n\nDay " + str(current_day) + ":")
    f.write("\n\tActual Reading: ")
    f.write("\n\t\tS1: " + str(s1) + " S2: " + str(s2) + " S3: "  + str(s3) + " S4: " + str(s4) + " S5: " + str(s5) + " S6: " + str(s6))
    f.write("\n\tValue in Sim: ")
    f.write("\n\t\tS1: " + str(s1_sim) + " S2: " + str(s2_sim) + " S3: "  + str(s3_sim) + " S4: " + str(s4_sim) + " S5: " + str(s5_sim) + " S6: " + str(s6_sim))
    f.write("\n\tAbsolute Error: ")
    f.write("\n\t\tS1 Error: " + str(s1_error) + " S2 Error: " + str(s2_error) + " S3 Error: "  + str(s3_error) + " S4 Error: " + str(s4_error) + " S5 Error: " + str(s5_error) + " S6 Error: " + str(s6_error))
    f.write("\n\tPercent Error: ")
    f.write("\n\t\tS1 Error: " + str(s1_perror) + " S2 Error: " + str(s2_perror) + " S3 Error: "  + str(s3_perror) + " S4 Error: " + str(s4_perror) + " S5 Error: " + str(s5_perror) + " S6 Error: " + str(s6_perror))
    save_water_grid(water_grid, current_day, "right_after_watering")
    if current_day == max_days:
        s1_a, s1_pa = None, None
        s2_a, s2_pa = None, None
        s3_a, s3_pa = None, None
        s4_a, s4_pa = None, None
        s5_a, s5_pa = None, None
        s6_a, s6_pa = None, None
        if len(s1_avg) != 0:
            s1_a, s1_pa  = sum(s1_avg)/len(s1_avg), sum(s1_pavg)/len(s1_pavg)
        if len(s2_avg) != 0:
            s2_a, s2_pa  = sum(s2_avg)/len(s2_avg), sum(s2_pavg)/len(s2_pavg)
        if len(s3_avg) != 0:
            s3_a, s3_pa  = sum(s3_avg)/len(s3_avg), sum(s3_pavg)/len(s3_pavg)
        if len(s4_avg) != 0:
            s4_a, s4_pa  = sum(s4_avg)/len(s4_avg), sum(s4_pavg)/len(s4_pavg)
        if len(s5_avg) != 0:
            s5_a, s5_pa  = sum(s5_avg)/len(s5_avg), sum(s5_pavg)/len(s5_pavg)
        if len(s6_avg) != 0:
            s6_a, s6_pa  = sum(s6_avg)/len(s6_avg), sum(s6_pavg)/len(s6_pavg)
        f.write("\n\nAverage Absolute Error: ")
        f.write("\n\t\tS1: " + str(s1_a) + " S2: " + str(s2_a) + " S3: "  + str(s3_a) + " S4: " + str(s4_a) + " S5: " + str(s5_a) + " S6: " + str(s6_a))
        f.write("\nAverage Percent Error: ")
        f.write("\n\t\tS1: " + str(s1_pa) + " S2: " + str(s2_pa) + " S3: "  + str(s3_pa) + " S4: " + str(s4_pa) + " S5: " + str(s5_pa) + " S6: " + str(s6_pa)) 
    return m

def determine_current_gain(day):
    """ Determines average gain from the soil moisture sensors within a six hour window around the time associated with GARDEN_START_DATE for a single day.
    """
    s_list = [[], [], [], [], [], []]
    for i in range(-7200, 9000, 1800):
        s1, s2, s3, s4, s5, s6 = getDeviceReadings(day-1, i)
        s_list[0].append(s1)
        s_list[1].append(s2)
        s_list[2].append(s3)
        s_list[3].append(s4)
        s_list[4].append(s5)
        s_list[5].append(s6) 

    return np.mean([max(s_list[i]) - min(s_list[i]) for i in range(6) if SOIL_MOISTURE_SENSOR_ACTIVE[i] == True])

def determine_avg_gain(day):
    """ Determines average gain from the soil moisture sensors within a six hour window around the time associated with GARDEN_START_DATE for days until the current day.
    """
    prev_gains = []
    for i in range(day-5, day): #sliding 5 day mean for weather purposes
        prev_gains.append(determine_current_gain(i))
    return np.mean(prev_gains)

def determine_evap_rate(day):
    """ Determines the evaporation rate for days until the current day.
    """
    prev_gains = []
    for i in range(day-5, day): #sliding 5 day mean for weather purposes
        prev_gains.append(determine_current_gain(i))
    return np.mean(prev_gains)

def initial_water_value():
    """ Determines the initial soil moisture readings to institatie the garden.
    """
    if not any(SOIL_MOISTURE_SENSOR_ACTIVE):
        return [0.10, 0.01] #default values if no soil sensors are being used
    s1, s2, s3, s4, s5, s6 = getDeviceReadings(0)
    slist = [s1, s2, s3, s4, s5, s6]
    active_list = [slist[i] for i in range(6) if SOIL_MOISTURE_SENSOR_ACTIVE[i] == True]
    return [np.mean(active_list), np.std(active_list)]


