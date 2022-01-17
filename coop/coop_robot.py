from itertools import chain
import os,sys
import numpy as np
import logging
from typing import Tuple

from URKFSControl.URDriver.robot import RobotModel, UniversalRobot

from scipy.spatial.transform import Rotation as R

from spatialmath.base import *
import PyKDL as kdl
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf

class DualCoopModel:
    def __init__(self, coop_models: Tuple[RobotModel]):
        self.__coop_model = coop_models
        self.__size = len(self.__coop_model)
        if self.__size !=2:
            raise(RuntimeError('Current coop model does not support more than two robot models'))
        
    def relativeOrient(self, q_tuple:Tuple[np.array]) -> np.array:
        rot_left = self.__coop_model[0].rot(q_tuple[0])
        rot_right = self.__coop_model[1].rot(q_tuple[1])
        rot_relative = np.transpose(rot_right)*rot_left
        return rot_relative

    def absoluteOrient(self, q_tuple:Tuple[np.array]) -> np.array:
        rot_left = self.__coop_model[0].rot(q_tuple[0])
        rot_right = self.__coop_model[1].rot(q_tuple[1])
        
        return 0

    @property
    def size(self):
        return self.__size

