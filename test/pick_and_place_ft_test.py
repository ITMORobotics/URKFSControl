#!/usr/bin/env python3

import sys
import os
import time
from tkinter.tix import Tree
from zlib import Z_TREES
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

def down_direct_force_control(force: np.ndarray, z_force) -> np.ndarray:
    # Coefficients
    max_speed = 0.001
    kf = 0.005
    ktau = 0.05
    vel = np.zeros(6)
    k_p = -0.0001
    k_i = -0.00001
    k_d = -0.00001


    if not hasattr(down_direct_force_control, 'integral'):
        down_direct_force_control.integral_time = time.time()
        down_direct_force_control.integral = 0
        down_direct_force_control.last_err = 0

    # time
    i_new_time = time.time()
    delta_t = i_new_time - down_direct_force_control.integral_time
    down_direct_force_control.integral_time = i_new_time

    # error
    err_f = z_force - force[2]
    derr_f = (err_f - down_direct_force_control.last_err)/delta_t
    down_direct_force_control.last_err = err_f

    # PID
    sum_f = k_p*err_f
    sum_f = sum_f if abs(sum_f) < max_speed else max_speed * sign(float(sum_f))
    sum_d = k_d*derr_f
    down_direct_force_control.integral += k_i* delta_t * err_f

    dirv = DIR_F if abs(force[2]) < 10 else DIR_F*0

    vel[0] = kf*deadzone(force[0], 3)
    vel[1] = kf*deadzone(force[1], 3)
    vel[2] = sum_f + sum_d + dirv[2]
    vel[3] = ktau*deadzone(force[3], 0.02)
    vel[4] = ktau*deadzone(force[4], 0.02)
    vel[5] = ktau*deadzone(force[5], 0.02)

    return vel

def pick_object(robot: URDriver.UniversalRobot) -> None:
    open_gripper()
    robot.control.moveL(PICK_P + Z_A*0.1)
    time.sleep(0.5)
    robot.control.moveL(PICK_P)
    time.sleep(0.5)
    close_gripper()
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

def open_gripper():
    control_gripper(id=0, operation_type=110, speed=0, position=0)

def close_gripper():
    control_gripper(id=0, operation_type=121, speed=30, position=0)

def free_gripper():
    control_gripper(id=0, operation_type=100, speed=0, position=0)




def main():

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

    # Reset force sensor
    robot1.control.zeroFtSensor()
    robot1.update_state()

    # Pick and place object
    pick_object(robot1)
    place_object(robot1)

    # Reset force sensor
    robot1.control.zeroFtSensor()
    robot1.update_state()

    fdir = DIR
    z0 = 0
    z1 = 0
    dz = 0

    while True:
        start = time.time()

        # state
        robot1.update_state()
        send_joint_states(robot1.state.q)
        send_wrench(robot1.state.f)

        # get end effector transform
        p, angvec = robot_model.pose_angvec(robot1.state.q)
        z1 = z0
        z0 = p[2]
        dz = z0 - z1
        # print('z', z0, dz)

        # Use medial filter to get forces
        fe = np.array(median.apply_median(robot1.state.f)).flatten()
        jac = robot_model.jacobian(robot1.state.q)

        # control
        fdir = down_direct_force_control(fe, 20)

        if z0 > Z_TR and dz < 0:
            print
            # print('START REG')
        # fdir = direct_force_control(fe)

        # Calculate speed vector using Manipulator Jacobian
        jspeed = np.linalg.pinv(jac).dot(fdir)

        # Move manipulator
        robot1.control.speedJ(jspeed, acceleration, dt)

        end = time.time()
        duration = end - start
        if duration < dt:
            time.sleep(dt - duration)

        if rospy.is_shutdown():
            free_gripper()
            break



if __name__ == '__main__':

    current_filepath = os.path.dirname(os.path.abspath(__file__))
    urdf_filepath = os.path.join(current_filepath, '..', 'urdf_model', 'ur5e.urdf')

    # ROS
    rospy.init_node('node_name')
    urdf_description = rospy.get_param('/robot_description')
    robot_model = URDriver.RobotModel(urdf_filepath, 'base', 'tool0')

    # Publishers
    js_publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
    wr_publisher = rospy.Publisher('/wrench', WrenchStamped, queue_size=10)

    # Services
    rospy.wait_for_service('/control_gripper')
    control_gripper = rospy.ServiceProxy('/control_gripper', control)

    main()
