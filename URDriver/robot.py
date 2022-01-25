from abc import abstractmethod
import os,sys
import numpy as np
import logging
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

import spatialmath.base as spmb
import PyKDL as kdl

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .state import RobotState
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf

class UniversalRobot:

    """
    Class contains rtde communication functions and robot state:
        - joint angles
        - joint velocities
        - joint currents
        - joint torques
        - actual TCP force
    """

    def __init__(self, ip: str):
        self.__ip_addr = ip
        self.__is_ok = False

        self.__control = RTDEControlInterface(self.__ip_addr)
        self.__receive = RTDEReceiveInterface(self.__ip_addr)

        self.__state = RobotState()
        self.update_state()


    def __del__(self):
        self.__control.speedStop()

    @property
    def is_ok(self) -> bool:
        return self.__is_ok

    @property
    def control(self) -> RTDEControlInterface:
        return self.__control

    @property
    def state(self) -> RobotState:
        return self.__state

    @abstractmethod
    def update_state(self):
        self.__state.q      = np.array(self.__receive.getActualQ())
        self.__state.dq     = np.array(self.__receive.getActualQd())
        self.__state.i      = np.array(self.__receive.getActualRobotCurrent())
        self.__state.tau    = np.array(self.__receive.getActualMomentum())
        self.__state.f      = np.array(self.__receive.getActualTCPForce())

class RobotModel:

    """
    Class wraps python KDL bindings
    """

    def __init__(self, urdf_filename: str, base_link: str, tool_link: str):
        with open(urdf_filename, 'r', encoding='utf-8') as urdf_file:
            urdf_str = urdf_file.read()

        # Generate kinematic model for orocos_kdl
        (ok, self.__tree) = urdf.treeFromString(urdf_str)

        if not ok:
            raise RuntimeError('Tree is not valid')
        self.__chain = self.__tree.getChain(base_link, tool_link)
        logging.info('Created chain for robot model: %s', self.__chain)
        self.__num_of_joints = self.__chain.getNrOfJoints()

        self.__fk_posesolver    = kdl.ChainFkSolverPos_recursive(self.__chain)
        self.__fk_velsolver     = kdl.ChainFkSolverVel_recursive(self.__chain)
        self.__ik_velsolver     = kdl.ChainIkSolverVel_pinv(self.__chain)
        self.__ik_posesolver    = kdl.ChainIkSolverPos_NR(self.__chain, self.__fk_posesolver, self.__ik_velsolver)

        self.__jacsolver        = kdl.ChainJntToJacSolver(self.__chain)
        self.__djacsolver       = kdl.ChainJntToJacDotSolver(self.__chain)

        self.__jacobian = kdl.Jacobian(self.__num_of_joints)
        self.__ee_frame = kdl.Frame()
        self.__vel_frame = kdl.FrameVel()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        self.__jacsolver.JntToJac(to_jnt_array(q), self.__jacobian)
        return to_np_matrix(self.__jacobian, q.shape[0])

    def pose_angvec(self, q: np.ndarray) -> np.ndarray:
        T = self.__fk_posesolver.JntToCart(to_jnt_array(q), self.__ee_frame)
        p = to_np_matrix(self.__ee_frame.p, 3)
        rot = to_np_matrix(kdl.Rotation(self.__ee_frame.M), 3)
        angvec = spmb.tr2angvec(rot, unit='rad', check=False)
        return (p, angvec)

    def rot(self, q: np.ndarray) -> np.ndarray:
        T = self.__fk_posesolver.JntToCart(to_jnt_array(q), self.__ee_frame)
        rot = to_np_matrix(kdl.Rotation(self.__ee_frame.M), 3)
        return rot

    def twist(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        self.__fk_velsolver.JntToCart(to_jnt_array_vel(q, dq), self.__vel_frame)
        return to_np_matrix(self.__vel_frame.GetTwist(), 6)


def to_np_matrix(kdl_data, size: int) -> np.ndarray:
    if isinstance(kdl_data, (kdl.JntSpaceInertiaMatrix, kdl.Jacobian, kdl.Rotation)):
        out = np.zeros((size, size))
        for i in range(0, size):
            for j in range(0, size):
                out[i,j] = kdl_data[i,j]
        return out
    elif isinstance(kdl_data, (kdl.JntArray, kdl.JntArrayVel)):
        out = np.zeros(size)
        for i in range(0, size):
            out[i] = kdl_data[i]
        return out
    else:
        out = np.zeros(size)
        for i in range(0, size):
            out[i] = kdl_data[i]
        return out

def to_jnt_array(np_vector: np.ndarray) -> kdl.JntArray:
    size = np_vector.shape[0]
    ja = kdl.JntArray(size)
    for i in range(0, size):
        ja[i] = np_vector[i]
    return ja

def to_jnt_array_vel(q: np.ndarray, dq:np.ndarray) -> kdl.JntArrayVel:
    size = q.shape[0]
    jav = kdl.JntArrayVel(size)
    jav.q = to_jnt_array(q)
    jav.qdot = to_jnt_array(dq)
    return jav

def quaternioun(matrix: np.ndarray) -> np.ndarray:
    pass