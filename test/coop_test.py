import numpy as np
import sys, os
import PyKDL as kdl

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
from URDriver import robot

np.set_printoptions(precision=4)

def main():
    robot_model_left = robot.RobotModel('urdf_model/ur5e_left.urdf','world', 'tool0')
    robot_model_right = robot.RobotModel('urdf_model/ur5e_right.urdf','world', 'tool0')
    

if __name__ == "__main__":
    main()