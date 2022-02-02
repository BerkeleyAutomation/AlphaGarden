import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import argparse
import imutils
import math
import glob
from movement import batch_target_approach
from control import start, MyHandler, mount_xPruner, mount_yPruner, dismount_xPruner, dismount_yPruner, mount_nozzle, dismount_nozzle, photo
from thread import FarmBotThread
import pickle
import argparse
from urllib.request import urlopen
from datetime import datetime, timezone
import os
from time import gmtime, strftime
import time
from PIL import Image, ImageOps
import sys
import pickle as pkl

from farmbot import Farmbot, FarmbotToken 
import requests
import json
import wget



if __name__ == '__main__':
    # fb = FarmBotThread()
    # fb.update_action("photo", None)

    # time.sleep(10)
    # photo("./")


    raw_token = FarmbotToken.download_token("markt@berkeley.edu",
                                        "a5s6d7f8fdsas",
                                        "https://my.farm.bot")
    API_TOKEN = str(json.loads(raw_token)["token"]["encoded"])
    headers = {'Authorization': 'Bearer ' + API_TOKEN,
               'content-type': "application/json"}
    response = requests.get('https://my.farmbot.io/api/images', headers=headers)
    images = response.json()
    # print(images)
    imageurls = [images[i]['attachment_url'] for i in range(len(images))]
    imagetimes = [images[i]['created_at'] for i in range(len(images))]
    imagexs = [images[i]['meta']['x'] for i in range(len(images))]
    imageys = [images[i]['meta']['y'] for i in range(len(images))]
    imagezs = [images[i]['meta']['z'] for i in range(len(images))]
    newTime = {}
    for t in imagetimes:
        ret = t[:4] + t[5:7] + t[8:10] + t[11:13] + t[14:16] + t[17:19]
        newTime[t] = int(ret)
    mri = imagetimes.index(max(newTime)) # Most Recent index
    print(max(newTime))
    # cname = wget.download(imageurls[mri])
