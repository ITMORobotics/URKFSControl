import numpy as np
import sys, os
import PyKDL as kdl
from spatialmath import SE3

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
from URDriver import robot

target_tf = SE3(-0.15, -0.5, 0.35) @ SE3.Ry(0.0, 'rad')

np.set_printoptions(precision=4)

def main():
    q = np.array([0.0,-np.pi/2, np.pi/2, 0.0, -np.pi/2, 0.0])
    dq = np.array([0.1,0, 0, 0.0, 0, 0.0])
    robot_model = robot.RobotModel('urdf_model/ur5e_right.urdf','world', 'tool0')
    print("Jacobian: \n ", robot_model.jacobian(q))
    print("pose, angvec: \n ", robot_model.pose_angvec(q))
    print("twist: \n ", robot_model.twist(q, dq))
    print("IK solution: \n ", robot_model.nik_q(target_tf.A, np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])))

if __name__ == "__main__":
    main()