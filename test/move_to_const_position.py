#!/usr/bin/env python3

import sys
import os
import time
import PyKDL
import rospy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import URDriver
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf

import numpy as np

from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import median_filter as md

J_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

BASE = 'base_link'
TOOL = 'tool0'
NUM_JOINTS = 6

JOINTS  = np.array([np.pi/2, -np.pi/2 + np.pi/6 - 0.2, -np.pi/2, -np.pi/2 - np.pi/6 + 0.2, np.pi/2, 0])
print(JOINTS)

DIR     = np.array([0, 0, -0.01, 0, 0, 0])
DIR_F   = np.array([0, 0, -0.1, 0, 0, 0])*0.02
R_Z     = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

np.set_printoptions(precision=4, suppress=True)

median = md.MedianFilter(NUM_JOINTS, 16)

# ROS
rospy.init_node('node_name')

robot1 = URDriver.UniversalRobot('192.168.88.5')
robot1.control.moveJ(JOINTS)
