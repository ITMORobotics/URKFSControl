import numpy as np
import sys, os
import PyKDL as kdl
from spatialmath import SE3

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from coop import coop_robot
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
from URDriver import robot

target_tf = SE3(-0.15, -0.5, 0.35) @ SE3.Ry(0.0, 'rad')

np.set_printoptions(precision=4)

def main():
    robot_model_left = robot.RobotModel('urdf_model/ur5e_left.urdf','world', 'tool0')
    robot_model_right = robot.RobotModel('urdf_model/ur5e_right.urdf','world', 'tool0')
    coop_model = coop_robot.DualCoopModel((robot_model_left, robot_model_right))

    tf1 = target_tf.A
    tf2 = target_tf.A
    q1 = robot_model_left.nik_q(tf1, np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    q2 = robot_model_left.nik_q(tf2, np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]))
    
    jj = coop_model.relative_jacobian((q1, q2))
    print(jj)



if __name__ == "__main__":
    main()