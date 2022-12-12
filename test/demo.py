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

from read_json.read_json import read_poses_json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
import URDriver
from coop import coop_robot
from coop import executor


np.set_printoptions(precision=4)

filename = './test/read_json/poses_box_move_v0.json'

coop_state0 = coop_robot.CoopCartState()
coop_state0.build_from_SE3(SE3(read_poses_json(filename, 'pose_prepare', 'abs'), check=False), SE3(read_poses_json(filename, 'pose_prepare', 'rel'),check=False))

coop_state1 = coop_robot.CoopCartState()
coop_state1.build_from_SE3(SE3(read_poses_json(filename, 'pose_start', 'abs'), check=False), SE3(read_poses_json(filename, 'pose_start', 'rel'), check=False))

coop_state2 = coop_robot.CoopCartState()
coop_state2.build_from_SE3(SE3(read_poses_json(filename, 'pose_finish_up', 'abs'), check=False), SE3(read_poses_json(filename, 'pose_finish_up', 'rel'), check=False))

coop_state3 = coop_robot.CoopCartState()
coop_state3.build_from_SE3(SE3(read_poses_json(filename, 'pose_finish', 'abs'),check=False), SE3(read_poses_json(filename, 'pose_finish', 'rel'),check=False))

coop_state_btw = coop_robot.CoopCartState()
coop_state_btw.build_from_SE3(SE3(read_poses_json(filename, 'pose_btw', 'abs'),check=False), SE3(read_poses_json(filename, 'pose_btw', 'rel'),check=False))

def main():
    coop_system = executor.CoopSmartSystem('192.168.88.5', '192.168.88.6', 'urdf_model/ur5e_left.urdf', 'urdf_model/ur5e_right.urdf', 'tool0')
    # coop_system.close_gripper(('left', 'right'))
    # time.sleep(1.0)
    # coop_system.open_gripper(('left', 'right'))
    time.sleep(1.0)
    coop_system.p2p_cartmove_avoid(copy.deepcopy(coop_state0), 15.0, False)
    time.sleep(1.0)
    coop_system.p2p_cartmove_avoid(copy.deepcopy(coop_state1), 10.0, False)
    # # coop_system.close_gripper(('left', 'right'))
    # time.sleep(1.5)
    # coop_system.p2p_cart_handle_move(copy.deepcopy(coop_state0), 5.0, False)
    # coop_system.zeroFT()
    # coop_system.p2p_cart_handle_move(copy.deepcopy(coop_state_btw), 10.0, True)
    # coop_system.p2p_cart_handle_move(copy.deepcopy(coop_state2), 10.0, False)
    # # time.sleep(1.0)
    # coop_system.p2p_cart_handle_move(copy.deepcopy(coop_state3), 5.0, False)
    # # coop_system.open_gripper(('left', 'right'))
    # time.sleep(2.5)
    # coop_system.p2p_cartmove_avoid(copy.deepcopy(coop_state2), 5.0, False)
    # coop_system.zeroFT()
    # coop_system.p2p_cartmove_avoid(copy.deepcopy(coop_state0), 15.0, False)


if __name__ == "__main__":
    main()
