import numpy as np
import sys, os
import PyKDL as kdl

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
from URDriver import robot

np.set_printoptions(precision=4)

def main():
    q = np.array([0.0,-np.pi/2, np.pi/2, 0.0, -np.pi/2, 0.0])
    dq = np.array([0.1,0, 0, 0.0, 0, 0.0])
    robot_model = robot.RobotModel('urdf_model/ur5e_right.urdf','world', 'tool0')
    print("Jacobian: \n ", robot_model.jacobian(q))
    print("pose, angvec: \n ", robot_model.pose_angvec(q))
    print("twist: \n ", robot_model.twist(q, dq))

if __name__ == "__main__":
    main()