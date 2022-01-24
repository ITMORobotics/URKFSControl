#!/usr/bin/env python3

import sys
import os
import time
import PyKDL
import rospy

import spatialmath as spm
import spatialmath.base as spmb

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
DIR     = np.array([0, 0, -0.01, 0, 0, 0])
DIR_F   = np.array([0, 0, -0.1, 0, 0, 0])*0.02
R_Z     = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

np.set_printoptions(precision=4, suppress=True)

median = md.MedianFilter(NUM_JOINTS, 16)

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

robot1 = URDriver.UniversalRobot('192.168.88.5')
robot1.control.moveJ(JOINTS)

robot1.update_state()
robot1.control.zeroFtSensor()
robot1.update_state()


q = arr2kdl(PyKDL.JntArray, robot1.state.q)
jac_kdl.JntToJac(q, jacobian)
jac_arr = kdl2mat(jacobian)

fk_p_kdl.JntToCart(q, transform)


rot_6_0 = R_Z @ rot2mat(transform.M)
p_6_0 = R_Z @ np.array(list(transform.p))
transform_6_0 = spm.SE3(*p_6_0) @ spm.SE3(spmb.r2t(rot_6_0, check=True))
transform_camera = transform_6_0 @ spm.SE3(-0.0305, -0.10218, 0.03445)
p_object = np.array([-0.0184, 0.0132, 0.509, 1])


print(rot_6_0)
print(p_6_0)
print(transform_6_0)
print(transform_camera.A.dot(p_object))

PLACE_P = np.array([0.085, 0.429, 0.3, np.pi, 0, 0])
print(PLACE_P)
