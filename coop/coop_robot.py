from itertools import chain
import os,sys
import numpy as np
import logging
from typing import Tuple
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

import spatialmath.base as spmb

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
        self.__robots[0].control.stopScript()
        self.__robots[1].control.stopScript()

    def __del__(self):
        self.__robots[0].__del__()
        self.__robots[1].__del__()


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
        rel_jacob = np.concatenate((1.0/2*self.__coop_model[0].jacobian(q_tuple[0]), 1.0/2*self.__coop_model[1].jacobian(q_tuple[1])), axis=1)
        return rel_jacob

    def abs_rel_jacobian(self, q_tuple: Tuple[np.ndarray]) -> np.ndarray:
        return np.concatenate((self.absolute_jacobian(q_tuple), self.relative_jacobian(q_tuple)), axis=0)
        
    def relative_orient(self, q_tuple:Tuple[np.ndarray]) -> np.ndarray:
        rot_left = self.__coop_model[0].rot(q_tuple[0])
        rot_right = self.__coop_model[1].rot(q_tuple[1])
        rot_relative = rot_left*rot_right.T
        return rot_relative

    def absolute_orient(self, q_tuple:Tuple[np.ndarray]) -> np.ndarray:
        rot_left = self.__coop_model[0].rot(q_tuple[0])
        # rot_right = self.__coop_model[1].rot(q_tuple[1])
        # rot_relative = rot_right @ rot_left.T
        # angvec_relative = list(spmb.tr2angvec(rot_relative, unit='rad', check=False))
        # # print(angvec_relative)
        # angvec_relative[0] /= 2.0
        # # print(angvec_relative)
        # half_rot_relative = spmb.angvec2r(angvec_relative[0], angvec_relative[1])

        # rot_abs = rot_left @ half_rot_relative
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

    @property
    def size(self):
        return self.__size