import numpy as np
import scipy
import copy
import sys, os
import PyKDL as kdl
from typing import Tuple
import time
from spatialmath import SE3,SO3, Twist3
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from coop import coop_robot
from coop import executor

np.set_printoptions(precision=4)

coop_state1 = coop_robot.CoopCartState()
coop_state1.build_from_SE3(SE3(-0.15, -0.5, 0.35) @ SE3.Rx(-np.pi, 'rad'), SE3(0.0, -0.3, 0.0) @ SE3.Ry(0.0, 'rad'))

coop_state2 = coop_robot.CoopCartState()
coop_state2.build_from_SE3(SE3(0.15, -0.5, 0.5) @ SE3.Rx(-np.pi+0.1, 'rad'), SE3(0.0, -0.3, 0.0) @ SE3.Rz(0.0, 'rad'))

def main():
    coop_system = executor.CoopSmartSystem('192.168.88.5', '192.168.88.6', 'urdf_model/ur5e_left.urdf', 'urdf_model/ur5e_right.urdf', 'tool0')
    print("Abs: \n", coop_system.get_state().abs_tf)
    print("Rel: \n", coop_system.get_state().rel_tf)
    # coop_system.p2p_cartmove_avoid(coop_state1, 5.0, False)
    # time.sleep(2.0)
    # coop_system.p2p_cartmove_avoid(coop_state2, 5.0, False)

if __name__ == "__main__":
    main()