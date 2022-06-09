import numpy as np
import scipy
import sys, os
import PyKDL as kdl
from typing import Tuple
import time
from spatialmath import SE3,SO3, Twist3
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
from URDriver import robot
from coop import coop_robot
from URDriver import controller

import pandas as pd 

np.set_printoptions(precision=4)

dt = 0.02

target_coop_abs_ft =   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
target_coop_rel_ft =   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


coop_state_to = coop_robot.CoopCartState()
coop_state_to.build_from_SE3(SE3(-0.15, -0.5, 0.55) @ SE3.Ry(0.0, 'rad') @ SE3.Rx(np.pi, 'rad'), SE3(-0.3, 0.0, 0.0) @ SE3.Rz(0.0, 'rad'))

coop_state_2 = coop_robot.CoopCartState()
coop_state_2.build_from_SE3(SE3(0.15, -0.5, 0.35) @ SE3.Ry(0.0, 'rad') @ SE3.Rx(np.pi, 'rad'), SE3(-0.28, 0.0, 0.0) @ SE3.Rz(0.0, 'rad'))

coop_state_3 = coop_robot.CoopCartState()
coop_state_3.build_from_SE3(SE3(0.2, -0.6, 0.35) @ SE3.Ry(0.0, 'rad') @ SE3.Rx(np.pi, 'rad'), SE3(-0.3, 0.0, 0.0) @ SE3.Rz(0.0, 'rad'))


coop_P_matrix_full = scipy.linalg.block_diag(np.identity(3)*0.3, np.identity(3)*0.5, np.identity(3)*0.3, np.identity(3)*0.5)*3.0
coop_I_matrix_full = scipy.linalg.block_diag(np.identity(3)*0.01, np.identity(3)*0.02, np.identity(3)*0.01, np.identity(3)*0.02)*0.0005

coop_stiff_matrix = scipy.linalg.block_diag(np.identity(3)*0.005,np.identity(3)*0.1, -np.identity(3)*0.005, -np.identity(3)*0.1)

def main():
    sM = coop_robot.generate_square_selection_matrix(np.array([0,0,0,0,0,0,1,0,1,1,1,1]))
    robot_model_left = robot.RobotModel('urdf_model/ur5e_left.urdf','world', 'tool0')
    robot_model_right = robot.RobotModel('urdf_model/ur5e_right.urdf','world', 'tool0')
    coop_model = coop_robot.DualCoopModel((robot_model_left, robot_model_right))
    coop_ur = coop_robot.DualUniversalRobot('192.168.88.5', '192.168.88.6', dt)
    coop_controller = controller.CooperativeController(coop_model, coop_stiff_matrix, coop_P_matrix_full, coop_I_matrix_full, dt)
    # print(external_stiff_sM)
    ok = coop_ur.control[0].zeroFtSensor()
    ok &= coop_ur.control[1].zeroFtSensor()
    if not ok:
        raise(RuntimeError('Force torque connection was broken'))

    start_time = time.time()
    data = []
    data_row = go_to_pose(coop_state_to, coop_model, coop_ur, coop_controller, 10.0, sM)
    data.append(data_row)
    data_row = go_to_pose(coop_state_2, coop_model, coop_ur, coop_controller, 10.0, sM)
    data.append(data_row)
    pd.DataFrame(np.concatenate(data), columns = ['pax', 'pay', 'paz', 'prx', 'pry', 'prz', 'uax', 'uay', 'uaz', 'urx', 'ury', 'urz']).to_csv("output_data.csv")



def go_to_pose(to: coop_robot.CoopCartState, coop_model: coop_robot.DualCoopModel, coop_ur: coop_robot.DualUniversalRobot, coop_controller: controller.CooperativeController, finish_time: float, sM: np.ndarray):
    start_time = time.time()
    coop_ur.update_state()
    out = []
    while time.time()-start_time < finish_time:
        start_loop_time = time.time()
        coop_ur.update_state()
        current_coop_abs_pose1, current_coop_abs_orient1, current_coop_rel_pose1, current_coop_rel_orient1 = coop_model.cart_state((coop_ur.state[0].q, coop_ur.state[1].q)).to_pose_rot()
        target_coop_abs_pose1, target_coop_abs_orient1, target_coop_rel_pose1, target_coop_rel_orient1 = to.to_pose_rot()
        control_dq, control_twist = coop_controller.hybride_world_control(
            target_coop_abs_pose1,
            target_coop_abs_orient1,
            target_coop_rel_pose1,
            target_coop_rel_orient1,

            target_coop_abs_ft,
            target_coop_rel_ft,
            (coop_ur.state[0].q, coop_ur.state[1].q),
            (coop_ur.state[0].f, coop_ur.state[1].f),
            sM[0], sM[1]
        )
        # Send dq control to two robots
        print(control_dq)
        wpose = np.concatenate([current_coop_abs_pose1, current_coop_rel_pose1, control_twist[0:3], control_twist[6:9] ] ,axis=0)
        out.append(wpose)
        coop_ur.send_dq(control_dq)

        end_loop_time = time.time()
        duration = end_loop_time - start_loop_time
        if duration < dt:
            time.sleep(dt - duration)
    coop_ur.stop()
    coop_controller.reset()
    return out

if __name__ == "__main__":
    main()