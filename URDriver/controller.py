import numpy as np
import scipy
import sys, os
from typing import Tuple
import time

from spatialmath import SE3,SO3

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from coop import coop_robot

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class MPIController():
    def __init__(self, P: np.ndarray, I: np.ndarray, dt: float):
        self.__P = P
        self.__I = I
        self.__dt = dt
        self.__integral_value = np.zeros(self.__I.shape[0])
        self.__wind_up_max = np.ones(self.__I.shape[0])*0.8
    
    def reset(self):
        self.__integral_value = np.zeros(self.__I.shape)
    
    def u(self, err: np.ndarray):
        if err.shape[0] != self.__P.shape[0]:
            raise(RuntimeError("Invalid error shape"))
        nonlimit_integral = self.__I @ err*self.__dt + self.__integral_value
        # print(nonlimit_integral)
        abs_integral_value = np.minimum(self.__wind_up_max, np.abs(nonlimit_integral))
        self.__integral_value = np.multiply(abs_integral_value,  np.sign(nonlimit_integral) )
        # print(self.__integral_value)
        u = self.__P @ err + self.__integral_value
        # print(u)
        return u


class CooperativeController():
    def __init__(self, 
        coop_model:coop_robot.DualCoopModel,
        coop_stiff_matrix: np.ndarray,
        coop_P_matrix: np.ndarray, 
        coop_I_matrix: np.ndarray,
        dt: float
        ):
        self.__coop_P_matrix = coop_P_matrix
        self.__coop_I_matrix = coop_I_matrix
        self.__coop_stiff_matrix = coop_stiff_matrix
        self.__coop_model = coop_model
        self.__pi_control = MPIController(self.__coop_P_matrix, self.__coop_I_matrix, dt)
        

    def world_stiff_control(self,
        target_coop_rel_pose: np.ndarray,
        target_coop_rel_orient: np.ndarray,
        target_coop_abs_ft: np.ndarray,
        q: Tuple[np.ndarray],
        ft: Tuple[np.ndarray],
        selection_matrix_T: np.ndarray,
        selection_matrix_Y: np.ndarray
        ) -> np.ndarray:
        
        # Reasign state values
        q1,q2 = q
        ft1,ft2 = ft

        abs_rot = self.__coop_model.absolute_orient((q1,q2))
        block_abs_rot = scipy.linalg.block_diag(abs_rot, abs_rot) # kroneker product for creating block diagonal matrix
        rel_rot = self.__coop_model.relative_orient((q1,q2))

        block_rot = scipy.linalg.block_diag(block_abs_rot, abs_rot, np.identity(3))

        # Calculating absolute and relative errors for force, pose and orientation
        rel_pose_error = target_coop_rel_pose - abs_rot.T @ self.__coop_model.relative_pose((q1,q2))
        rel_orient_error = target_coop_rel_orient @ rel_rot.T

        rel_orient_error_tf = SE3(rel_pose_error) @ SE3(SO3(rel_orient_error, check=False))
        rel_orient_twist_error = rel_orient_error_tf.twist().A[3:]

        print(rel_orient_twist_error)

        abs_ft_error = target_coop_abs_ft - block_abs_rot.T @ self.__coop_model.absolute_force_torque((ft1,ft2))

        error_move_coop = np.concatenate((rel_pose_error, rel_orient_twist_error), axis=0)
        error_force_coop = abs_ft_error

        target_move_twist = block_rot @ selection_matrix_T @  self.__pi_control.u(error_move_coop)
        target_force_twist = block_rot @ selection_matrix_Y @ self.__coop_stiff_matrix @ -error_force_coop
        # print("Target move twist: \n", target_move_twist)
        # print("rel pose: \n", coop_model.relative_pose((q1,q2)) )
        control_dq = np.zeros(q1.shape)
        abs_rel_jacob = self.__coop_model.abs_rel_jacobian((q1,q2))
        jacob_rank = np.linalg.matrix_rank(abs_rel_jacob, 1e-5)
        if jacob_rank < abs_rel_jacob.shape[0]:
            print("Given jacobian is singular")
            control_dq = np.zeros(q1.shape)
        else:
            control_dq = np.linalg.pinv(self.__coop_model.abs_rel_jacobian((q1,q2))) @ (target_move_twist + target_force_twist)
            # print("Control dq: \n", control_dq)
        return control_dq

    def world_rigid_control(self,
        target_coop_abs_pose: np.ndarray,
        target_coop_abs_orient: np.ndarray,
        target_coop_rel_pose: np.ndarray,
        target_coop_rel_orient: np.ndarray,
        q: Tuple[np.ndarray],
        ft: Tuple[np.ndarray],
        selection_matrix_T: np.ndarray,
        selection_matrix_Y: np.ndarray
        ) -> np.ndarray:
        
        # Reasign state values
        q1,q2 = q

        abs_rot = self.__coop_model.absolute_orient((q1,q2))
        block_abs_rot = scipy.linalg.block_diag(np.identity(3), np.identity(3)) # kroneker product for creating block diagonal matrix
        rel_rot = self.__coop_model.relative_orient((q1,q2))

        block_rot = scipy.linalg.block_diag(block_abs_rot, abs_rot, np.identity(3))

        # Calculating absolute  errors for  pose and orientation
        abs_pose_error = target_coop_abs_pose - self.__coop_model.absolute_pose((q1,q2))
        abs_orient_error = target_coop_abs_orient @ abs_rot.T

        abs_orient_error_tf = SE3(abs_pose_error) @ SE3(SO3(abs_orient_error, check=False))
        abs_orient_twist_error = abs_orient_error_tf.twist().A[3:]

        # Calculating relative errors for pose and orientation
        rel_pose_error = target_coop_rel_pose - abs_rot.T @ self.__coop_model.relative_pose((q1,q2))
        rel_orient_error = target_coop_rel_orient @ rel_rot.T

        rel_orient_error_tf = SE3(rel_pose_error) @ SE3(SO3(rel_orient_error, check=False))
        rel_orient_twist_error = rel_orient_error_tf.twist().A[3:]

        print(abs_orient_twist_error)

        error_move_coop = np.concatenate((abs_pose_error, abs_orient_twist_error, rel_pose_error, rel_orient_twist_error), axis=0)

        target_move_twist = block_rot @ selection_matrix_T @ self.__pi_control.u(error_move_coop)
        # print("Target move twist: \n", target_move_twist)
        # print("rel pose: \n", coop_model.relative_pose((q1,q2)) )
        control_dq = np.zeros(q1.shape)
        abs_rel_jacob = self.__coop_model.abs_rel_jacobian((q1,q2))
        jacob_rank = np.linalg.matrix_rank(abs_rel_jacob, 1e-5)
        if jacob_rank < abs_rel_jacob.shape[0]:
            print("Given jacobian is singular")
            control_dq = np.zeros(q1.shape)
        else:
            control_dq = np.linalg.pinv(self.__coop_model.abs_rel_jacobian((q1,q2))) @ (target_move_twist)
            # print("Control dq: \n", control_dq)
        return control_dq