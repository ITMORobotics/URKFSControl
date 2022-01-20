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

# JOINTS  = np.array([0, -np.pi/2, -np.pi/2, -np.pi, 0, 0])
JOINTS  = np.array([0, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0])
DIR     = np.array([0, 0, -0.01, 0, 0, 0])
DIR_F   = np.array([0, 0, -0.1, 0, 0, 0])
R_Z     = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

np.set_printoptions(precision=4, suppress=True)

median = md.MedianFilter(6, 19)

# ROS
rospy.init_node('node_name')
urdf_description = rospy.get_param('/robot_description')
# with open('urdf_model/ur5e_right.urdf', 'r', encoding='utf-8') as f:
#     urdf_description = f.read()

# Publishers
js_publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
wr_publisher = rospy.Publisher('/wrench', WrenchStamped, queue_size=10)



def arr2kdl(type, arr: np.ndarray):
    kdl_arr = type(NUM_JOINTS)
    for i in range(len(arr)):
        kdl_arr[i] = arr[i]
    return kdl_arr

def kdl2mat(kdl) -> np.ndarray:
    mat = np.zeros((kdl.columns(), kdl.rows()))
    for i in range(kdl.columns()):
        for j in range(kdl.rows()):
            mat[i, j] = kdl[i, j]
    return mat

def rot2mat(kdl) -> np.ndarray:
    mat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mat[i, j] = kdl[i, j]
    return mat


def deadzone(val, level):
    out = val if abs(val) > level else 0
    return out

def direct_force_control(force: np.ndarray) -> np.ndarray:
    kf = 0.001
    ktau = 0.6
    vel = np.zeros(6)
    vel[0] = kf*deadzone(force[0], 10)
    vel[1] = kf*deadzone(force[1], 10)
    vel[2] = kf*deadzone(force[2], 10)
    vel[3] = ktau*deadzone(force[3], 0.25)
    vel[4] = ktau*deadzone(force[4], 0.25)
    vel[5] = ktau*deadzone(force[5], 0.25)

    return vel

# KDL
print(urdf_description)
ok, tree = urdf.treeFromString(urdf_description)
arm_chain = tree.getChain(BASE, TOOL)

# KDL Solvers
fk_p_kdl    = PyKDL.ChainFkSolverPos_recursive(arm_chain)
fk_v_kdl    = PyKDL.ChainFkSolverVel_recursive(arm_chain)
ik_v_kdl    = PyKDL.ChainIkSolverVel_pinv(arm_chain)
ik_p_kdl    = PyKDL.ChainIkSolverPos_NR(arm_chain, fk_p_kdl, ik_v_kdl)
jac_kdl     = PyKDL.ChainJntToJacSolver(arm_chain)
dyn_kdl     = PyKDL.ChainDynParam(arm_chain, PyKDL.Vector.Zero())

# KDL usage
pos_kdl_array = PyKDL.JntArray(NUM_JOINTS)
vel_kdl_array = PyKDL.JntArrayVel(NUM_JOINTS)

jacobian = PyKDL.Jacobian(NUM_JOINTS)
transform = PyKDL.Frame()
jv = JOINTS

wrench = WrenchStamped()
state = JointState()
state.name = J_NAMES


# Parameters
velocity = 0.5
acceleration = 0.5
dt = 1.0/500  # 2ms
lookahead_time = 0.1
gain = 300

robot1 = URDriver.UniversalRobot('192.168.88.6')
robot1.control.moveJ(JOINTS)
robot1.update_state()

fdir = DIR


for i in range(100000):
    start = time.time()

    # state
    robot1.update_state()
    state.header = Header()
    state.header.stamp = rospy.Time.now()
    state.position = robot1.state.q
    js_publisher.publish(state)


    q = arr2kdl(PyKDL.JntArray, robot1.state.q)
    jac_kdl.JntToJac(q, jacobian)
    jac_arr = kdl2mat(jacobian)

    fk_p_kdl.JntToCart(q, transform)
    rot_6_0 = rot2mat(transform.M)
    # print(rot_6_0)

    fe = np.concatenate((
        R_Z @ robot1.state.f[0:3],
        R_Z @ robot1.state.f[3:],
    ))
    fe = median.apply_median(fe)
    wrench.header = Header()
    wrench.header.stamp = rospy.Time.now()
    wrench.wrench.force.x   = fe[0]
    wrench.wrench.force.y   = fe[1]
    wrench.wrench.force.z   = fe[2]
    wrench.wrench.torque.x  = fe[3]
    wrench.wrench.torque.y  = fe[4]
    wrench.wrench.torque.z  = fe[5]
    wr_publisher.publish(wrench)

    # control
    fdir = direct_force_control(fe)

    # jv = robot1.state.q + np.linalg.pinv(jac_arr).dot(fdir)
    jspeed = np.linalg.pinv(jac_arr).dot(fdir)
    # fk_p_kdl.JntToCart(q, transform)
    # print(transform.p)

    robot1.control.speedJ(jspeed, acceleration, dt)
    # robot1.control.servoJ(jv, velocity, acceleration, dt, lookahead_time, gain)

    end = time.time()
    duration = end - start
    # print(duration)
    if duration < dt:
        time.sleep(dt - duration)

    if rospy.is_shutdown():
        break
