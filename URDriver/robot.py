from itertools import chain
import os,sys
import numpy as np
import logging
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from .state import RobotState

from spatialmath.base import *
import PyKDL as kdl
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf

class UniversalRobot:

    def __init__(self, ip: str):
        self.__ip_addr = ip
        self.__is_ok = False

        self.__control = RTDEControlInterface(self.__ip_addr)
        self.__receive = RTDEReceiveInterface(self.__ip_addr)

        self.__state = RobotState()
        self.update_state()

    @property
    def is_ok(self) -> bool:
        return self.__is_ok

    @property
    def control(self) -> RTDEControlInterface:
        return self.__control

    @property
    def state(self) -> RobotState:
        return self.__state

    def update_state(self):
        self.__state.q      = np.array(self.__receive.getActualQ())
        self.__state.dq     = np.array(self.__receive.getActualQd())
        self.__state.i      = np.array(self.__receive.getActualRobotCurrent())
        self.__state.tau    = np.array(self.__receive.getActualMomentum())
        self.__state.f      = np.array(self.__receive.getActualTCPForce())

class RobotModel:
    def __init__(self, urdf_filename: str, base_link: str, tool_link: str):
        urdf_file = open(urdf_filename,'r')
        urdf_str = urdf_file.read()
        urdf_file.close()
        # Generate kinematic model for orocos_kdl
        (ok, self.__tree) = urdf.treeFromString(urdf_str)
        
        if not ok:
            raise RuntimeError('Tree is not valid')
        self.__chain = self.__tree.getChain(base_link, tool_link)
        logging.info("Created chain for robot model: %s", self.__chain)

        self.__fk_posesolver = kdl.ChainFkSolverPos_recursive(self.__chain)
        self.__fk_velsolver    = kdl.ChainFkSolverVel_recursive(self.__chain)
        self.__ik_velsolver    = kdl.ChainIkSolverVel_pinv(self.__chain)
        self.__ik_posesolver    = kdl.ChainIkSolverPos_NR(self.__chain, self.__fk_posesolver, self.__ik_velsolver)
        
        self.__jacsolver = kdl.ChainJntToJacSolver(self.__chain)
        self.__djacsolver = kdl.ChainJntToJacDotSolver(self.__chain)
        
    def jacobian(self, q: np.array) -> np.array:
        jac = kdl.Jacobian(q.shape[0])
        self.__jacsolver.JntToJac(to_jnt_array(q), jac)
        return to_np_matrix(jac, q.shape[0])
    
    def pose_angvec(self, q: np.array) -> np.array:
        end_frame = kdl.Frame()
        T = self.__fk_posesolver.JntToCart(to_jnt_array(q), end_frame)
        p = to_np_matrix(end_frame.p,3)
        rot = to_np_matrix(kdl.Rotation(end_frame.M),3)
        angvec = tr2angvec(rot, unit='rad', check=False)
        translate = p
        return (p, angvec)
    
    def twist(self, q: np.array, dq: np.array) -> np.array:
        vel_frame = kdl.FrameVel()
        self.__fk_velsolver.JntToCart(to_jnt_array_vel(q, dq), vel_frame)
        return to_np_matrix(vel_frame.GetTwist(),6)

    def jacobian_dot(self, q: np.array, dq: np.array):
        pass

def to_np_matrix(kdl_data, size: int) -> np.array:
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

def to_jnt_array(np_vector: np.array)-> kdl.JntArray:
    size = np_vector.shape[0]
    ja = kdl.JntArray(size)
    for i in range(0, size):
        ja[i] = np_vector[i]
    return ja

def to_jnt_array_vel(q: np.array, dq:np.array)-> kdl.JntArrayVel:
    size = q.shape[0]
    jav = kdl.JntArrayVel(size)
    jav.q = to_jnt_array(q)
    jav.qdot = to_jnt_array(dq)
    return jav


# def toJA(np_matrix):
#     size = np.shape(np_matrix)[0]
#     ja = kdl.JntArray(size)
#     for i in range(0, size):
#         ja[i] = np_matrix[i]
#     return ja
    

# def Ta(theta, psi):
#     c_psi = cos(psi)
#     s_psi = sin(psi)
#     c_theta = cos(theta)
#     s_theta = sin(theta)
#     Ta = np.matrix([ [1, 0, 0, 0, 0, 0],
#                      [0, 1, 0, 0, 0, 0],
#                      [0, 0, 1, 0, 0, 0],
#                      [0, 0, 0, c_psi*s_theta, -s_psi, 0],
#                      [0, 0, 0, s_psi*s_theta, c_psi, 0],
#                      [0, 0, 0, c_theta, 0, 1]])
#     return Ta

# def xa(q):
#     end_frame = kdl.Frame()
#     rs.fksolver.JntToCart(toJA(q), end_frame)
#     p = end_frame.p
#     rot = kdl.Rotation(end_frame.M)
#     zyz = rot.GetRPY()
#     xa_vec = np.concatenate((get_np_matrix(p,3), get_np_matrix(zyz,3)), axis=0)
#     return xa_vec.T[0]


# def jaca(current_q):
#     end_frame = kdl.Frame()
#     rs.fksolver.JntToCart(toJA(current_q), end_frame)
#     p = end_frame.p
#     rot = kdl.Rotation(end_frame.M)
#     zyz = rot.GetRPY()

#     return inv(Ta(zyz[1], zyz[2])).dot(jac(current_q))

# def jac(current_q):
#     jac = kdl.Jacobian(6)
#     print(jac)
#     rs.jacsolver.JntToJac(toJA(current_q), jac)
#     return get_np_matrix(jac,6)

# def djac(current_q, current_dq):
#     current_qdq = kdl.JntArrayVel(current_q, current_dq)
#     djac = kdl.Jacobian(6)
#     kdl.JntToJacDot(current_qdq, djac)
#     return get_np_matrix(djac, 6)