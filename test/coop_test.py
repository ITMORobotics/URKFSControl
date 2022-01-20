from tracemalloc import start
import numpy as np
import sys, os
import PyKDL as kdl
import time


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
from URDriver import robot
from coop import coop_robot

np.set_printoptions(precision=4)

start_time = time.time()
dt = 0.05

def main():
    robot_model_left = robot.RobotModel('urdf_model/ur5e_left.urdf','world', 'tool0')
    robot_model_right = robot.RobotModel('urdf_model/ur5e_right.urdf','world', 'tool0')
    coop_model = coop_robot.DualCoopModel((robot_model_left, robot_model_right))
    coop_ur = coop_robot.DualUniversalRobot('192.168.88.5', '192.168.88.6')
    while time.time()-start_time < 30.0:
        start_loop_time = time.time()
        coop_ur.update_state()
        q1 = coop_ur.state[0].q
        q2 = coop_ur.state[1].q
        ft1 = coop_ur.state[0].f
        ft2 = coop_ur.state[1].f
        # q1 = np.array([0.0, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, np.pi/2.0, 0.0])
        # q2 = np.array([np.pi, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0, np.pi/2.0, 0.0])
        # print("abs jacob: \n ", coop_model.absolute_jacobian((q1, q2)))
        # print("rel jacob: \n", coop_model.relative_jacobian((q1, q2)))

        # print("abs orient: \n", coop_model.absolute_orient((q1, q2)))
        # print("rel orient: \n ", coop_model.relative_orient((q1, q2)))
        print("abs pose: \n", coop_model.absolute_pose((q1,q2)) )
        print("rel pose: \n", coop_model.relative_pose((q1,q2)) )
        print("abs ft: \n", coop_model.absolute_force((ft1,ft2)) )
        print("rel ft: \n", coop_model.relative_force((ft1,ft2)) )
        
        end_loop_time = time.time()

        duration = end_loop_time - start_loop_time
        if duration < dt:
            time.sleep(dt - duration)



if __name__ == "__main__":
    main()