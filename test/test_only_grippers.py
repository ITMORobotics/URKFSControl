import serial
import sys
import os
import glob
import logging
import time
from math import pi
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gripper_controller import GripperSerialController
import URDriver

# s1 = serial.Serial('/dev/gripper_left')
gripper_left = GripperSerialController('/dev/gripper_left', 57600)
gripper_right = GripperSerialController('/dev/gripper_right', 57600)

gripper_left.open()
gripper_right.open()
time.sleep(1.0)
gripper_left.close()
gripper_right.close()
time.sleep(1.0)
gripper_left.open()
gripper_right.open()