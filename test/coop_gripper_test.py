import numpy as np
import scipy
import sys, os
import glob, serial
import PyKDL as kdl
from typing import Tuple
import time
import copy
from spatialmath import SE3,SO3, Twist3
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
import URDriver
from coop import coop_robot
from coop import executor

from gripper_controller import GripperSerialController


np.set_printoptions(precision=4)

T0_ABS = np.array([[  0.9924,   0.1188, -0.031730, -0.2446],
                   [  0.1186,  -0.9929, -0.009534, -0.5202],
                   [-0.03264, 0.005699,   -0.9995,  0.4097],
                   [       0,        0,         0,       1]])

T1_ABS = np.array([[  0.08178,  -0.9964, -0.02269,  0.4742],
                   [  -0.9935, -0.08332,  0.07796, -0.1821],
                   [ -0.07957,  0.01617,  -0.9967,  0.2466],
                   [        0,         0,       0,       1]])

T_REL = np.array([[ 1, 0, 0,  -0.30265],
                  [ 0, 1, 0 ,   0.0224],
                  [ 0, 0, 1,   0.03887],
                  [ 0, 0, 0,         1]])

coop_state1 = coop_robot.CoopCartState()
coop_state1.build_from_SE3(SE3(T0_ABS, check=False), SE3(T_REL,check=False))

coop_state2 = coop_robot.CoopCartState()
coop_state2.build_from_SE3(SE3(T1_ABS,check=False), SE3(T_REL,check=False))

gripper_left = GripperSerialController('/dev/gripper_left', 57600)
gripper_right = GripperSerialController('/dev/gripper_right', 57600)

def main():
    coop_system = executor.CoopSmartSystem('192.168.88.5', '192.168.88.6', 'urdf_model/ur5e_left.urdf', 'urdf_model/ur5e_right.urdf', 'tool0')
    gripper_left.open()
    gripper_right.open()
    time.sleep(2.0)
    coop_system.p2p_cartmove_avoid(copy.deepcopy(coop_state1), 15.0, False)
    time.sleep(2.0)
    gripper_left.close()
    gripper_right.close()
    time.sleep(2.0)
    coop_system.p2p_cartmove_avoid(copy.deepcopy(coop_state2), 15.0, False)
    time.sleep(2.0)
    gripper_left.open()
    gripper_right.open()
    time.sleep(2.0)

if __name__ == "__main__":
    main()
