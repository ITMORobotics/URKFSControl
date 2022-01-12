import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from .state import RobotState

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