#!/usr/bin/env python3


import sys
import os
import time
import spatialmath as spm
import spatialmath.base as spmb
import PyKDL
import rospy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import URDriver
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
import median_filter as md

from gripper_pkg.srv import control

J_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

BASE = 'base_link'
TOOL = 'tool0'
NUM_JOINTS = 6

EYE = np.eye(3, 3)

JOINTS  = np.array([np.pi/2, -np.pi/2 + np.pi/6 - 0.2, -np.pi/2, -np.pi/2 - np.pi/6 + 0.2, np.pi/2, 0])
PICK_P  = np.array([-0.4, 0, 0.216, np.pi, 0, 0])
# PLACE_P = np.array([0.085, 0.429, 0.0481, np.pi, 0, 0])
# [0.0896 0.4143 0.0468 1.    ]
# PLACE_P = np.array([0.0885, 0.4154, 0.0481, np.pi, 0, 0])
PLACE_P = np.array([0.0851, 0.4255, 0.0472, np.pi, 0, 0])

DIR     = np.array([0, 0, -0.01, 0, 0, 0])
DIR_F   = np.array([0, 0, -0.1, 0, 0, 0])*0.01
R_Z     = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
Z_A     = np.array([0, 0, 1, 0, 0, 0])

Z_TR    = 0.28
# CAMERA_TRANSFORM = spm.SE3(-0.0305, -0.10218, 0.03445)
CAMERA_TRANSFORM = np.array([-0.0305, -0.10218, 0.03445])

# Stages
# 0 -- touch
# 1 -- search
# 2 -- insert
STAGE = 0

np.set_printoptions(precision=4, suppress=True)



def deadzone(val, level):
    out = val if abs(val) > level else 0
    return out

def sign(val: float) -> float:
    return (val > 0) - (val < 0)

def pick_object(robot: URDriver.UniversalRobot) -> None:
    robot.control.moveL(PICK_P + Z_A*0.1)
    time.sleep(0.5)
    robot.control.moveL(PICK_P)
    time.sleep(0.5)
    robot.control.moveL(PICK_P + Z_A*0.1)
    time.sleep(0.5)

def place_object(robot: URDriver.UniversalRobot) -> None:
    robot.control.moveL(PLACE_P + Z_A*0.4)
    time.sleep(0.5)
    robot.control.moveL(PLACE_P + Z_A*0.4)

def send_joint_states(q: np.ndarray):
    state = JointState()
    state.name = J_NAMES
    state.header = Header()
    state.header.stamp = rospy.Time.now()
    state.position = q
    js_publisher.publish(state)

def send_wrench(f: np.ndarray):
    wrench = WrenchStamped()
    wrench.header = Header()
    wrench.header.stamp     = rospy.Time.now()
    wrench.wrench.force.x   = f[0]
    wrench.wrench.force.y   = f[1]
    wrench.wrench.force.z   = f[2]
    wrench.wrench.torque.x  = f[3]
    wrench.wrench.torque.y  = f[4]
    wrench.wrench.torque.z  = f[5]
    wr_publisher.publish(wrench)

def set_orientation(zd: np.ndarray, z2: np.ndarray, angle: float):
    vel = np.zeros(6)

    diff = -1*(zd - z2)
    w = np.cross(z2, diff)

    vel[3:] = w
    return (np.linalg.norm(w) < 1e-4), vel

def rpy2rv(roll, pitch, yaw):

    alpha = yaw
    beta = pitch
    gamma = roll

    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)

    r11 = ca*cb
    r12 = ca*sb*sg-sa*cg
    r13 = ca*sb*cg+sa*sg
    r21 = sa*cb
    r22 = sa*sb*sg+ca*cg
    r23 = sa*sb*cg-ca*sg
    r31 = -sb
    r32 = cb*sg
    r33 = cb*cg

    theta = np.arccos((r11+r22+r33-1)/2)
    sth = np.sin(theta)
    kx = (r32-r23)/(2*sth)
    ky = (r13-r31)/(2*sth)
    kz = (r21-r12)/(2*sth)

    return [(theta*kx),(theta*ky),(theta*kz)]

def skew(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def esimate(G: np.ndarray, x_n: np.ndarray, fi: np.ndarray, F: np.ndarray):
    G = G - G @ fi @ np.linalg.inv(np.eye(3, 3) + fi.T @ G @ fi) @ fi.T @ G
    x_n = x_n - G @ fi @ (fi.T @ x_n - F)

    return G, x_n

def load_f_regressor() -> np.ndarray:
    return EYE

def load_t_regressor(f: np.ndarray, rot_6_0: np.ndarray) -> np.ndarray:
    return -skew(f) @ rot_6_0

def load_parameters_estimation(f: np.ndarray, regr: np.ndarray) -> np.ndarray:
    pass

def main():

    global PLACE_P

    # Initialize median filter
    median = md.MedianFilter(NUM_JOINTS, 17)

    # Parameters
    velocity = 0.0001
    acceleration = 0.0001
    dt = 1.0/500  # 2ms
    lookahead_time = 0.1
    gain = 300

    # Initialize robot
    robot1 = URDriver.UniversalRobot('192.168.88.5')

    # Move to initial position
    robot1.control.moveJ(JOINTS)
    # time.sleep(5)

    robot1.update_state()

    # Pick and place object
    place_object(robot1)

    robot1.update_state()

    # Get actual PLACE_P + Z_A*0.4 Rotation matrix
    rot_6_0 = robot_model.rot(robot1.state.q)
    ee = spm.SO3(rot_6_0)

    new_rot = spm.SO3.Rx(np.pi/6) @ ee
    place = np.copy(PLACE_P + Z_A*0.4)
    place[3:] = rpy2rv(*new_rot.rpy())
    robot1.control.moveL(place)


    params = np.zeros(3)
    h = np.eye(3, 3)

    array = []

    while True:

        start = time.time()

        # state
        robot1.update_state()

        send_wrench(robot1.state.f)
        rot_6_0 = spm.SO3(robot_model.rot(robot1.state.q))
        fe = np.array(median.apply_median(robot1.state.f)).flatten()
        # array.append(np.concatenate((robot1.state.q, fe)))

        # Control
        # Sprial movement
        new_rot = spm.SO3.Rz(0.001) @ new_rot
        xx = np.cross(ee.o, new_rot.a)
        rot_n = spm.SE3.OA(np.cross(new_rot.a, xx), new_rot.a)
        place[3:] = rpy2rv(*rot_n.rpy())

        f_regr = load_f_regressor()
        h, params = esimate(h, params, f_regr, fe[:3])

        aaa = np.zeros(6)
        aaa[:3] = params
        # send_wrench(params)

        # Move manipulator
        robot1.control.servoL(place, velocity, acceleration, dt, lookahead_time, gain)

        end = time.time()
        duration = end - start
        if duration < dt:
            time.sleep(dt - duration)

        if rospy.is_shutdown():
            # df = pd.DataFrame(array)
            # df.to_csv('ft_sensor.csv', index=False)
            break


if __name__ == '__main__':

    current_filepath = os.path.dirname(os.path.abspath(__file__))
    urdf_filepath = os.path.join(current_filepath, '..', 'urdf_model', 'ur5e_fc.urdf')

    # ROS
    rospy.init_node('online_ft_calibrator')
    # robot_model = URDriver.RobotModel(urdf_filepath, 'base', 'obj')
    robot_model = URDriver.RobotModel(urdf_filepath, 'base', 'tool0')

    # Publishers
    js_publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
    wr_publisher = rospy.Publisher('/wrench', WrenchStamped, queue_size=10)

    main()
