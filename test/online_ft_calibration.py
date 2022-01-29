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

def direct_force_control(force: np.ndarray) -> np.ndarray:
    kf = 0.001
    ktau = 0.1
    vel = np.zeros(6)
    vel[0] = kf*deadzone(force[0], 3)
    vel[1] = kf*deadzone(force[1], 3)
    vel[2] = kf*deadzone(force[2], 3)
    vel[3] = ktau*deadzone(force[3], 0.25)
    vel[4] = ktau*deadzone(force[4], 0.25)
    vel[5] = ktau*deadzone(force[5], 0.25)

    return vel

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
    robot.control.moveL(PLACE_P + Z_A*0.27)

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

def load_regressor(q: np.ndarray, rot_6_0: np.ndarray) -> np.ndarray:
    pass

def load_parameters_estimation(f: np.ndarray, regr: np.ndarray) -> np.ndarray:
    pass

def main():

    global PLACE_P

    # Initialize median filter
    median = md.MedianFilter(NUM_JOINTS, 5)

    # Parameters
    velocity = 0.1
    acceleration = 0.1
    dt = 1.0/500  # 2ms
    lookahead_time = 0.1
    gain = 300

    # Initialize robot
    robot1 = URDriver.UniversalRobot('192.168.88.5')

    # Move to initial position
    robot1.control.moveJ(JOINTS)
    time.sleep(5)

    # Reset force sensor
    robot1.control.zeroFtSensor()
    robot1.update_state()


    circ_msg = rospy.wait_for_message('/circle_pose', Pose, timeout=None)
    print(circ_msg)

    circ_p = np.array([circ_msg.position.x, circ_msg.position.y, 0, 1])
    p_6, _ = robot_model.pose_angvec(robot1.state.q)
    r_6    = robot_model.rot(robot1.state.q)
    transform_6_0 = spm.SE3(*p_6) @ spm.SE3(spmb.r2t(r_6, check=True))
    transform_camera = transform_6_0 @ spm.SE3(-0.0305, -0.10218, 0.03445)
    circ_p = transform_camera.A.dot(circ_p)
    PLACE_P[0] = circ_p[0]
    PLACE_P[1] = circ_p[1]


    # Pick and place object
    place_object(robot1)

    # Reset force sensor
    robot1.control.zeroFtSensor()
    robot1.update_state()

    fdir = DIR

    while True:
        start = time.time()

        # state
        robot1.update_state()
        send_joint_states(robot1.state.q)

        # Use medial filter to get forces
        fe = np.array(median.apply_median(robot1.state.f)).flatten()
        jac = robot_model.jacobian(robot1.state.q)

        # send_wrench(robot1.state.f)
        rot_6_0 = robot_model.rot(robot1.state.q)

        # Control
        # Sprial movement

        # Parameters estimation



        # Calculate speed vector using Manipulator Jacobian
        jspeed = np.linalg.pinv(jac).dot(fdir)

        # Move manipulator
        robot1.control.speedJ(jspeed, acceleration, dt)

        end = time.time()
        duration = end - start
        if duration < dt:
            time.sleep(dt - duration)

        if rospy.is_shutdown():
            break


if __name__ == '__main__':

    current_filepath = os.path.dirname(os.path.abspath(__file__))
    urdf_filepath = os.path.join(current_filepath, '..', 'urdf_model', 'ur5e_fc.urdf')

    # ROS
    rospy.init_node('online_ft_calibrator')
    robot_model = URDriver.RobotModel(urdf_filepath, 'base', 'obj')

    # Publishers
    js_publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
    wr_publisher = rospy.Publisher('/wrench', WrenchStamped, queue_size=10)

    main()
