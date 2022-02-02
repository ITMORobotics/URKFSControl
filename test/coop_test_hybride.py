import numpy as np
import scipy
import sys, os
import PyKDL as kdl
from typing import Tuple
import time
from spatialmath import SE3,SO3, Twist3
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
from URDriver import robot
from coop import coop_robot
from URDriver import controller

np.set_printoptions(precision=4)

dt = 0.02

target_coop_abs_ft =   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
target_coop_rel_ft =   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


coop_state_to = coop_robot.CoopCartState()
coop_state_to.build_from_SE3(SE3(-0.15, -0.5, 0.35) @ SE3.Ry(0.0, 'rad') @ SE3.Rx(np.pi, 'rad'), SE3(0.0, -0.3, 0.0) @ SE3.Rz(0.0, 'rad'))

coop_P_matrix_full = scipy.linalg.block_diag(np.identity(3)*0.3, np.identity(3)*0.5, np.identity(3)*0.3, np.identity(3)*0.5)*15.0
coop_I_matrix_full = scipy.linalg.block_diag(np.identity(3)*0.01, np.identity(3)*0.02, np.identity(3)*0.01, np.identity(3)*0.02)*0.001

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

    coop_ur.update_state()

    while time.time()-start_time < 50.0:
        start_loop_time = time.time()
        coop_ur.update_state()

        target_coop_abs_pose1, target_coop_abs_orient1, target_coop_rel_pose1, target_coop_rel_orient1 = coop_state_to.to_pose_rot()
        control_dq = coop_controller.hybride_world_control(
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
        coop_ur.send_dq(control_dq)

        end_loop_time = time.time()
        duration = end_loop_time - start_loop_time
        if duration < dt:
            time.sleep(dt - duration)


if __name__ == "__main__":
    main()