from tracemalloc import start
import numpy as np
import scipy
import sys, os
import PyKDL as kdl
from typing import Tuple
import time
from scipy.spatial.transform import Rotation as R


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
from URDriver import robot
from coop import coop_robot

np.set_printoptions(precision=4)

start_time = time.time()
dt = 0.02

target_coop_abs_ft =   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

target_coop_rel_pose = np.array([0.0, 0.0, 0.3])
target_coop_rel_orient = R.from_euler('x', np.pi/4, degrees=False).as_matrix()
print(target_coop_rel_orient)

coop_P_matrix = scipy.linalg.block_diag(np.identity(3)*0.0015, np.identity(3)*0.0015, np.identity(3)*0.25, np.identity(3)*3.0)
coop_stiff_matrix = scipy.linalg.block_diag(np.identity(3)*0.002, np.identity(3)*0.02, np.identity(3)*0.002, np.identity(3)*0.04)


def generate_simple_selection_matrix(allow_moves: np.ndarray) ->Tuple[np.ndarray, np.ndarray]:
    
    T_list = []
    Y_list = []

    T_columns = np.argwhere(allow_moves==1).flatten()
    Y_columns = np.argwhere(allow_moves==0).flatten()

    for i in range(0, T_columns.shape[0]):
        T_nonzero_column = np.zeros((allow_moves.shape[0],1))
        T_nonzero_column[T_columns[i] , 0] = 1.0
        T_list.append(T_nonzero_column)

    for i in range(0, Y_columns.shape[0]):
        Y_nonzero_column = np.zeros((allow_moves.shape[0],1))
        Y_nonzero_column[Y_columns[i] , 0] = 1.0
        Y_list.append(Y_nonzero_column)
    
    result_matrix_T = np.concatenate(T_list, axis=1)
    result_matrix_Y = np.concatenate(Y_list, axis=1)
    # print(result_matrix_T)
    # print(result_matrix_Y)
    return (result_matrix_T, result_matrix_Y)

def world_stiff_control(
    q: Tuple[np.ndarray],
    ft: Tuple[np.ndarray],
    coop_model:coop_robot.DualCoopModel,
    selection_matrix_T: np.ndarray,
    selection_matrix_Y: np.ndarray
    ) -> np.ndarray:
    
    # Reasign state values
    q1,q2 = q
    ft1,ft2 = ft

    abs_rot = coop_model.absolute_orient((q1,q2))
    block_abs_rot = scipy.linalg.block_diag(abs_rot, abs_rot) # kroneker product for makeing block diagonal matrix
    rel_rot = coop_model.relative_orient((q1,q2))

    block_rot = scipy.linalg.block_diag(block_abs_rot, block_abs_rot)

    # Calculating absolute and relative errors for force, pose and orientation
    rel_pose_error = target_coop_rel_pose - abs_rot.T @ coop_model.relative_pose((q1,q2))
    rel_orient_error_quat = R.from_matrix(target_coop_rel_orient @ rel_rot.T).as_quat()

    rel_orient_error = np.zeros(3)
    rel_orient_error = rel_orient_error_quat[:3] * rel_orient_error_quat[3]
    print(rel_orient_error)

    abs_ft_error = target_coop_abs_ft - block_abs_rot.T @ coop_model.absolute_force_torque((ft1,ft2))

    error_move_coop = np.concatenate((rel_pose_error, rel_orient_error), axis=0)
    error_force_coop = abs_ft_error

    target_move_twist = coop_P_matrix @ block_rot @ selection_matrix_T @ error_move_coop
    target_force_twist = coop_stiff_matrix @ block_rot @ selection_matrix_Y @ -error_force_coop
    # print("Target move twist: \n", target_move_twist)
    # print("rel pose: \n", coop_model.relative_pose((q1,q2)) )
    control_dq = np.zeros(q1.shape)
    abs_rel_jacob = coop_model.abs_rel_jacobian((q1,q2))
    jacob_rank = np.linalg.matrix_rank(abs_rel_jacob, 1e-5)
    if jacob_rank < abs_rel_jacob.shape[0]:
        print("Given jacobian is singular")
        control_dq = np.zeros(q1.shape)
    else:
        control_dq = np.linalg.pinv(coop_model.abs_rel_jacobian((q1,q2))) @ (target_move_twist + target_force_twist)
        # print("Control dq: \n", control_dq)
    return control_dq
    

def main():
    external_stiff_sM = generate_simple_selection_matrix(np.array([0,0,0,0,0,0,1,1,1,1,1,1]))
    robot_model_left = robot.RobotModel('urdf_model/ur5e_left.urdf','world', 'tool0')
    robot_model_right = robot.RobotModel('urdf_model/ur5e_right.urdf','world', 'tool0')
    coop_model = coop_robot.DualCoopModel((robot_model_left, robot_model_right))
    coop_ur = coop_robot.DualUniversalRobot('192.168.88.5', '192.168.88.6', dt)
    # print(external_stiff_sM)
    ok = coop_ur.control[0].zeroFtSensor()
    ok &= coop_ur.control[1].zeroFtSensor()
    if not ok:
        raise(RuntimeError('Force torque connection was broken'))

    while time.time()-start_time < 30.0:
        start_loop_time = time.time()
        coop_ur.update_state()

        control_dq = world_stiff_control(
            (coop_ur.state[0].q, coop_ur.state[1].q),
            (coop_ur.state[0].f, coop_ur.state[1].f),
            coop_model, external_stiff_sM[0], external_stiff_sM[1])

        # Send dq control to two robots
        coop_ur.send_dq(control_dq)

        end_loop_time = time.time()
        duration = end_loop_time - start_loop_time
        if duration < dt:
            time.sleep(dt - duration)


def check(q1,q2, coop_model):
    q1 = np.array([0.0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, np.pi/2.0, 0.0])
    q2 = np.array([np.pi, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, np.pi/2.0, 0.0])
    print("abs jacob: \n ", coop_model.absolute_jacobian((q1, q2)))
    print("rel jacob: \n", coop_model.relative_jacobian((q1, q2)))

    print("abs orient: \n", coop_model.absolute_orient((q1, q2)))
    print("rel orient: \n ", coop_model.relative_orient((q1, q2)))
    print("abs pose: \n", coop_model.absolute_pose((q1,q2)) )
    print("rel pose: \n", coop_model.relative_pose((q1,q2)) )


if __name__ == "__main__":
    main()