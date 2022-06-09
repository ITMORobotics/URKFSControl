from itertools import chain
import os,sys
import numpy as np
import logging
from typing import Tuple
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import spatialmath.base as spmb
from spatialmath import SE3, SO3
import roboticstoolbox

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from URDriver.robot import RobotModel, UniversalRobot, RobotState

from spatialmath.base import *
import PyKDL as kdl
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf


class DualUniversalRobot(UniversalRobot):
    def __init__(self, ip1: str, ip2: str, dt: float):
        self.__acceleration = 0.5
        self.__dt = dt
        self.__robots = (UniversalRobot(ip1), UniversalRobot(ip2) )
        self.__state = (RobotState(), RobotState())
        self.update_state()

    @property
    def is_ok(self) -> bool:
        return self.__is_ok

    @property
    def control(self) -> Tuple[RTDEControlInterface]:
        return ( self.__robots[0].control, self.__robots[1].control )

    @property
    def state(self) -> Tuple[RobotState]:
        return self.__state

    def update_state(self):
        self.__robots[0].update_state()
        self.__robots[1].update_state()
        self.__state = (self.__robots[0].state, self.__robots[1].state)
    
    def send_dq(self, dq_common):
        self.__robots[0].control.speedJ(dq_common[0:6], self.__acceleration, self.__dt); 
        self.__robots[1].control.speedJ(dq_common[6:], self.__acceleration, self.__dt) 
    
    def stop(self):
        self.__robots[0].control.speedStop()
        self.__robots[1].control.speedStop()
        # self.__robots[0].control.stopScript()
        # self.__robots[1].control.stopScript()

    def __del__(self):
        self.__robots[0].__del__()
        self.__robots[1].__del__()

class CoopCartState():
    def __init__(self):
        self.__abs_tf = None
        self.__rel_tf = None
    
    def build_from_SE3(self, abs_tf: SE3, rel_tf: SE3):
        self.__abs_tf = abs_tf
        self.__rel_tf = rel_tf
    
    def build_from_matrices(self, abs_tf: np.ndarray, rel_tf: np.ndarray):
        self.__abs_tf = SE3(abs_tf, check=False)
        self.__rel_tf = SE3(rel_tf, check=False)
        
    def to_matrices(self) -> Tuple[np.ndarray]:
        return self.__abs_tf.A, self.__rel_tf.A

    def to_pose_rot(self) -> Tuple[np.ndarray]:
        return (self.__abs_tf.t, self.__abs_tf.R, self.__rel_tf.t, self.__rel_tf.R)
    
    @property
    def rel_tf(self):
        return self.__rel_tf
    
    @property
    def abs_tf(self):
        return self.__abs_tf

class DualCoopModel:
    def __init__(self, coop_models: Tuple[RobotModel]):
        self.__coop_model = coop_models
        self.__size = len(self.__coop_model)
        if self.__size !=2:
            raise(RuntimeError('Current coop model does not support more than two robot models'))
    
    def relative_jacobian(self, q_tuple: Tuple[np.ndarray]) -> np.ndarray:
        rel_jacob = np.concatenate((-self.__coop_model[0].jacobian(q_tuple[0]), self.__coop_model[1].jacobian(q_tuple[1])), axis=1)
        return rel_jacob
    
    def absolute_jacobian(self, q_tuple: Tuple[np.ndarray]) -> np.ndarray:
        rel_jacob = np.concatenate((-1.0/2*self.__coop_model[0].jacobian(q_tuple[0]), 1.0/2*self.__coop_model[1].jacobian(q_tuple[1])), axis=1)
        return rel_jacob

    def abs_rel_jacobian(self, q_tuple: Tuple[np.ndarray]) -> np.ndarray:
        return np.concatenate((self.absolute_jacobian(q_tuple), self.relative_jacobian(q_tuple)), axis=0)
        
    def relative_orient(self, q_tuple:Tuple[np.ndarray]) -> np.ndarray:
        rot_left = self.__coop_model[0].rot(q_tuple[0])
        rot_right = self.__coop_model[1].rot(q_tuple[1])
        rot_relative = rot_right @ rot_left.T
        return rot_relative

    def absolute_orient(self, q_tuple:Tuple[np.ndarray]) -> np.ndarray:
        rot_left = self.__coop_model[0].rot(q_tuple[0])
        rot_abs = rot_left 
        return rot_abs
    
    def absolute_pose(self, q_tuple:Tuple[np.ndarray]) -> np.ndarray:
        pose_left = self.__coop_model[0].pose_angvec(q_tuple[0])[0]
        pose_right = self.__coop_model[1].pose_angvec(q_tuple[1])[0]
        # print(pose_right)
        return 1/2.0*(pose_left + pose_right)
    
    def relative_pose(self, q_tuple:Tuple[np.ndarray]) -> np.ndarray:
        # print("---")
        pose_left = self.__coop_model[0].pose_angvec(q_tuple[0])[0]
        # print(pose_left)
        pose_right = self.__coop_model[1].pose_angvec(q_tuple[1])[0]
        # print(pose_right)
        return pose_right - pose_left
    
    def relative_force_torque(self, ft: Tuple[np.ndarray]) -> np.ndarray:
        return ft[1] - ft[0]
    
    def absolute_force_torque(self, ft: Tuple[np.ndarray]) -> np.ndarray:
        return 1/2.0*(ft[0] + ft[1])
    
    def abs_rel_pose(self, q_tuple:Tuple[np.ndarray]) -> np.ndarray:
        return np.concatenate((self.absolute_pose(q_tuple), self.relative_pose(q_tuple)), axis=0)
    
    def abs_rel_force(self, ft:Tuple[np.ndarray]) -> np.ndarray:
        return np.concatenate((self.absolute_force(ft), self.relative_force(ft)), axis=0)

    def cart_state(self, q_tuple:Tuple[np.ndarray]) -> CoopCartState:
        state = CoopCartState()
        pose_abs = self.absolute_pose(q_tuple)
        rot_abs = self.absolute_orient(q_tuple)
        pose_rel = rot_abs.T @ self.relative_pose(q_tuple)
        rot_rel = self.relative_orient(q_tuple)
        
        frame_abs = SE3(pose_abs[0], pose_abs[1], pose_abs[2]) @ SE3(SO3(trnorm(rot_abs)))
        frame_rel = SE3(pose_rel[0], pose_rel[1], pose_rel[2]) @ SE3(SO3(trnorm(rot_rel)))
        # print(frame_rel)
        state.build_from_SE3(frame_abs, frame_rel)
        return state

    @property
    def size(self):
        return self.__size
    
class SE3LineTrj():
    def __init__(self, init_frame: SE3, finish_frame: SE3, dt: float, finish_time: float):
        self.__init_frame = init_frame
        self.__finish_frame = finish_frame
        self.__dt = dt
        self.__finish_time = finish_time
        self.__N =  int(self.__finish_time/self.__dt)

        key_rots = R.from_matrix(np.array([self.__init_frame.R, self.__finish_frame.R]))
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        times = list(np.arange(0, 1.0, 1.0/int(self.__finish_time/self.__dt)))
        # print(times)
        self.__interp_rots = slerp(times)
        self.__interp_pose = self.interpLine(self.__init_frame.t, self.__finish_frame.t, self.__N)
    
    def interpLine(self, p1: np.array, p2: np.array, N: int)-> list:
        vectors = []
        for i in range(0, self.__N):
            s = i*1.0/N
            vectors.append((1.0 - s) * p1 + s * p2)
        return vectors
    
    def getSE3(self, t: float) -> SE3:
        index = int(t/self.__dt)
        if index<=0:
            return self.__init_frame
        elif index>=self.__N:
            return self.__finish_frame
        else:
            # while True:
            #     if(not self.__trj[index].t.any()):
            #         index-=1
            #     else:
            vec = self.__interp_pose[index]
            tfvec = SE3(vec[0], vec[1], vec[2])
            tf_out = tfvec @ SE3(SO3(self.__interp_rots[index].as_matrix()), check=False)
            # print(vec)
            return tf_out

class CoopSE3LineTrj():
    def __init__(self, start_coop_state: CoopCartState, finish_coop_state: CoopCartState, dt: float, finish_time: float):
        self.__abs_trj = SE3LineTrj(start_coop_state.abs_tf, finish_coop_state.abs_tf, dt, finish_time)
        self.__rel_trj = SE3LineTrj(start_coop_state.rel_tf, finish_coop_state.rel_tf, dt, finish_time)
    
    def getState(self, time: float) -> CoopCartState:
        trj_state = CoopCartState()
        trj_state.build_from_SE3(self.__abs_trj.getSE3(time), self.__rel_trj.getSE3(time))
        # print(trj_state.rel_tf)
        return trj_state

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
    
    if len(T_list)<1:
        result_matrix_T = None
    else:
        result_matrix_T = np.concatenate(T_list, axis=1)
        
    if len(Y_list)<1:
        result_matrix_Y = None
    else:
        result_matrix_Y = np.concatenate(Y_list, axis=1)
    return (result_matrix_T, result_matrix_Y)


def generate_square_selection_matrix(allow_moves: np.ndarray) ->Tuple[np.ndarray, np.ndarray]:
    T_matrix = np.diag(allow_moves)
    Y_matrix = np.identity(T_matrix.shape[0])-T_matrix
    return (T_matrix, Y_matrix)