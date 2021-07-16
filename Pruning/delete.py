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

if __name__ == '__main__':
	fb = FarmBotThread()
	fb.update_action('read_pin', 8)