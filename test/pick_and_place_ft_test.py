#!/usr/bin/env python3

from glob import glob
import sys
import os
import time
import spatialmath as spm
import spatialmath.base as spmb
import PyKDL
import rospy
import math

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import URDriver
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf

import numpy as np
import numpy.linalg as LA

from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
import median_filter as md

from gripper_pkg.srv import control

J_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

BASE = 'base_link'
# TOOL = 'tool0'
# TOOL = 'rigid_gripper'
TOOL = 'obj'
NUM_JOINTS = 6


JOINTS  = np.array([np.pi/2, -np.pi/2 + np.pi/6 - 0.2, -np.pi/2, -np.pi/2 - np.pi/6 + 0.2, np.pi/2, 0])
PICK_P  = np.array([-0.4, 0, 0.216, np.pi, 0, 0])
# PLACE_P = np.array([0.085, 0.429, 0.0481, np.pi, 0, 0])
# [0.0896 0.4143 0.0468 1.    ]
# PLACE_P = np.array([0.0885, 0.4154, 0.0481, np.pi, 0, 0])

PLACE_P = np.array([0.0851, 0.4255, 0.0472, np.pi, 0, 0])
# PLACE_P = np.array([0.0851, 0.4255, 0.0352, np.pi, 0, 0])


DIR     = np.array([0, 0, -0.01, 0, 0, 0])
DIR_F   = np.array([0, 0, -0.1, 0, 0, 0])*0.01
R_Z     = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
Z_A     = np.array([0, 0, 1, 0, 0, 0])

Z_TR    = 0.28
# CAMERA_TRANSFORM = spm.SE3(-0.0305, -0.10218, 0.03445)
CAMERA_TRANSFORM = np.array([-0.0305, -0.10218, 0.03445])

# Stages
# 0 -- touch
# 1 -- search
# 2 -- insert
STAGE = 0

np.set_printoptions(precision=4, suppress=True)



def deadzone(val, level):
    out = val if abs(val) > level else 0
    return out

def direct_force_control(force: np.ndarray) -> np.ndarray:
    kf = 0.001
    ktau = 0.1
    vel = np.zeros(6)
    vel[0] = kf*deadzone(force[0], 3)
    vel[1] = kf*deadzone(force[1], 3)
    vel[2] = kf*deadzone(force[2], 3)
    vel[3] = ktau*deadzone(force[3], 0.25)
    vel[4] = ktau*deadzone(force[4], 0.25)
    vel[5] = ktau*deadzone(force[5], 0.25)

    return vel

def sign(val: float) -> float:
    return (val > 0) - (val < 0)

def down_direct_force_control(force: np.ndarray, z_force) -> np.ndarray:
    # Coefficients
    max_speed = 0.001
    kf = 0.05
    ktau = 0.005
    vel = np.zeros(6)
    k_p = -0.0001
    k_i = -0.00001
    k_d = -0.00001


    if not hasattr(down_direct_force_control, 'integral'):
        down_direct_force_control.integral_time = time.time()
        down_direct_force_control.integral = 0
        down_direct_force_control.last_err = 0

    # time
    i_new_time = time.time()
    delta_t = i_new_time - down_direct_force_control.integral_time
    down_direct_force_control.integral_time = i_new_time

    # error
    err_f = z_force - force[2]
    derr_f = (err_f - down_direct_force_control.last_err)/delta_t
    down_direct_force_control.last_err = err_f

    # PID
    sum_f = k_p*err_f
    sum_f = sum_f if abs(sum_f) < max_speed else max_speed * sign(float(sum_f))
    sum_d = k_d*derr_f
    down_direct_force_control.integral += k_i* delta_t * err_f

    dirv = DIR_F if abs(force[2]) < 10 else DIR_F*0

    vel[0] = kf*deadzone(force[0], 3)
    vel[1] = kf*deadzone(force[1], 3)
    vel[2] = sum_f + sum_d + dirv[2]
    vel[3] = -ktau*deadzone(force[3], 0.02)
    vel[4] = -ktau*deadzone(force[4], 0.02)
    vel[5] = ktau*deadzone(force[5], 0.02)

    return vel

def down_odirect_force_control(force: np.ndarray, z_force, rot_6_0: np.ndarray) -> np.ndarray:
    # Coefficients
    max_speed = 0.001

    kf = 0.01/2
    ktau = 0.1

    vel = np.zeros(6)
    f_dzone = np.zeros(6)
    k_p = -0.0001
    k_i = -0.00001
    k_d = -0.00001

    alpha_c = np.pi/12

    if not hasattr(down_odirect_force_control, 'integral'):
        down_odirect_force_control.integral_time = time.time()
        down_odirect_force_control.integral = 0
        down_odirect_force_control.last_err = 0
        down_odirect_force_control.last_f_6 = np.array([0,0,0,0,0,0])


    # time
    i_new_time = time.time()
    delta_t = i_new_time - down_odirect_force_control.integral_time
    down_odirect_force_control.integral_time = i_new_time

    f_6 = rot_6_0.T.dot(force[:3])
    t_6 = rot_6_0.T.dot(force[3:])

    ee = spm.SE3(spm.SO3(rot_6_0))
    hole = spm.SE3.Rx(np.pi)
    diff = ee.delta(hole)

    data = np.zeros(6)
    data[:3] = force[:3]
    data[3:] = t_6
    send_wrench(data)

    # error
    goal = -z_force
    err_f = goal - f_6[2]
    derr_f = (err_f - down_odirect_force_control.last0_err)/delta_t
    d_f_6 = (f_6 - down_odirect_force_control.last_f_6)/delta_t
    down_odirect_force_control.last_err = err_f
    down_odirect_force_control.last_f_6 = f_6


    # PID
    sum_f = k_p*err_f
    sum_f = sum_f if abs(sum_f) < max_speed else max_speed * sign(float(sum_f))
    sum_d = k_d*derr_f
    down_odirect_force_control.integral += k_i* delta_t * err_f

    dirv = -DIR_F if abs(f_6[2]) < 10 else DIR_F*0

    omega = diff[3:]
    k_omega = 0.1

    f_6[0] = kf*deadzone(f_6[0], 3) + k_d*d_f_6[0]
    f_6[1] = kf*deadzone(f_6[1], 3) + k_d*d_f_6[1]
    f_6[2] = sum_f + sum_d + dirv[2]
    t_6[0] = ktau*deadzone(t_6[0], 0.04) + k_omega*omega[0]
    t_6[1] = ktau*deadzone(t_6[1], 0.04) + k_omega*omega[1]
    t_6[2] = ktau*deadzone(t_6[2], 0.04) + k_omega*omega[2]
    vel[:3] = rot_6_0.dot(f_6)
    vel[3:] = rot_6_0.dot(t_6)

    print('err', err_f, derr_f)

    return vel

def find_object_force_control(force: np.ndarray, z_force, rot_6_0: np.ndarray) -> np.ndarray:

    global STAGE

    # Coefficients
    max_speed = 0.001
    kf = 0.01
    ktau = 0.01
    vel = np.zeros(6)
    f_dzone = np.zeros(6)
    k_p = -0.0001
    k_i = -0.00001
    k_d = -0.00001

    if not hasattr(down_odirect_force_control, 'integral'):
        down_odirect_force_control.integral_time = time.time()
        down_odirect_force_control.integral = 0
        down_odirect_force_control.last_err = 0

    # time
    i_new_time = time.time()
    delta_t = i_new_time - down_odirect_force_control.integral_time
    down_odirect_force_control.integral_time = i_new_time

    f_6 = rot_6_0.T.dot(force[:3])

    data = np.zeros(6)
    data[:3] = force[:3]
    send_wrench(data)

    # error
    goal = -z_force
    err_f = goal - f_6[2]
    derr_f = (err_f - down_odirect_force_control.last_err)/delta_t
    down_odirect_force_control.last_err = err_f

    # PID
    sum_f = k_p*err_f
    sum_f = sum_f if abs(sum_f) < max_speed else max_speed * sign(float(sum_f))
    sum_d = k_d*derr_f
    down_odirect_force_control.integral += k_i* delta_t * err_f

    dirv = -DIR_F if abs(f_6[2]) < 10 else DIR_F*0

    f_6[0] = 0
    f_6[1] = 0
    f_6[2] = sum_f + sum_d + dirv[2]
    vel[:3] = rot_6_0.dot(f_6)

    # print('err', err_f, derr_f)
    if np.abs(err_f) < 1:
        STAGE = 1


    return vel


def detect_hole_force_control(force: np.ndarray, z_force: float) -> np.ndarray:

    global STAGE

    # Coefficients
    max_speed = 0.001
    max_torque = 0.04

    vel = np.zeros(6)
    k_p = -0.0001
    k_i = -0.00001
    k_d = -0.00001

    period = 6
    n = 3

    if not hasattr(detect_hole_force_control, 'integral'):
        detect_hole_force_control.start_time = time.time()
        detect_hole_force_control.integral_time = time.time()
        detect_hole_force_control.integral = 0
        detect_hole_force_control.last_err = 0

        detect_hole_force_control.forces_dict = {}
        #минимальные и максимальные значения на каждой итерации
        detect_hole_force_control.max_force_iteration = 0
        detect_hole_force_control.max_force_direction_iteration = 0
        detect_hole_force_control.min_force_iteration = 100
        detect_hole_force_control.min_force_direction_iteration = 0


        detect_hole_force_control.direction_list = []
        detect_hole_force_control.last_rot_num = -1
        detect_hole_force_control.stage_delay_init = 0
    # time
    i_new_time = time.time()
    delta_t = i_new_time - detect_hole_force_control.integral_time
    detect_hole_force_control.integral_time = i_new_time

    # error
    err_f = z_force - force[2]
    derr_f = (err_f - detect_hole_force_control.last_err) / delta_t
    detect_hole_force_control.last_err = err_f

    # PID
    sum_f = k_p * err_f
    sum_f = sum_f if abs(sum_f) < max_speed else max_speed * sign(float(sum_f))
    sum_d = k_d * derr_f
    detect_hole_force_control.integral += k_i * delta_t * err_f

    angle_to_rotate = (time.time() - detect_hole_force_control.start_time) / period*2*math.pi

    data = np.zeros(6)
    data[:3] = force[:3]
    data[3:] = force[3:]


    # анализ силы по оси z
    detected_force = force[2]
    # отклонение
    if time.time() - detect_hole_force_control.start_time < period/4:
        vel[3] = max_torque * math.cos(angle_to_rotate)
    # Вращение
    elif time.time() - detect_hole_force_control.start_time < period*n:
        current_rot_num = angle_to_rotate // (2 * math.pi)
        #Инициализация минимума и максимума
        if len(detect_hole_force_control.direction_list) < current_rot_num+1:
            detect_hole_force_control.direction_list.append(0)
            print("update")
            sum_angle = 0
            counter = 0
            print()
            for key in detect_hole_force_control.forces_dict:
                if detect_hole_force_control.forces_dict[key] < ((detect_hole_force_control.max_force_iteration+detect_hole_force_control.min_force_iteration)/3):
                    print(key)
                    if key < detect_hole_force_control.max_force_direction_iteration:
                        sum_angle += key+math.pi*2
                    else:
                        sum_angle += key
                    counter += 1
            if counter > 0:
                print("ang:_"+ str((sum_angle/counter)%(math.pi*2)))
                detect_hole_force_control.direction_list[int(current_rot_num)]=(sum_angle/counter)
                print(detect_hole_force_control.direction_list[int(current_rot_num)])

            detect_hole_force_control.forces_dict = {}
            detect_hole_force_control.min_force_iteration = 100
            detect_hole_force_control.min_force_direction_iteration = 0
            detect_hole_force_control.max_force_iteration = 0
            detect_hole_force_control.max_force_direction_iteration = 0
            # Вращение
        vel[3] = max_torque * math.cos(angle_to_rotate)
        vel[4] = max_torque * math.sin(angle_to_rotate)
        data[1] = angle_to_rotate
        # проверка минимумов и максимумов только после первого вращения
        if (time.time() - detect_hole_force_control.start_time > period):
            detect_hole_force_control.forces_dict[angle_to_rotate] = detected_force
        #минимальная сила на итерации
            if detected_force < detect_hole_force_control.min_force_iteration:
                detect_hole_force_control.min_force_direction_iteration = angle_to_rotate
                detect_hole_force_control.min_force_iteration = detected_force
        # максимальная сила на итерациии
            if detected_force > detect_hole_force_control.max_force_iteration:
                detect_hole_force_control.max_force_direction_iteration = angle_to_rotate
                detect_hole_force_control.max_force_iteration = detected_force

    #перемещение в направлении отверстия
    else:

        if detect_hole_force_control.stage_delay_init == 0:
            print(detect_hole_force_control.direction_list)
            detect_hole_force_control.stage_delay_init = time.time()
        vel[0] = max_speed * 2 * math.cos(detect_hole_force_control.direction_list[2]+math.pi-math.pi/15)
        vel[1] = max_speed * 2 * math.sin(detect_hole_force_control.direction_list[2]+math.pi-math.pi/15)
        dirv = DIR_F if abs(force[2]) < 10 else DIR_F * 0
        vel[2] = sum_f + sum_d + dirv[2]
        if (force[2] < 7 or force[1] > 20 or force[0] > 20) and time.time()-detect_hole_force_control.stage_delay_init > 2:
            STAGE = 2
    send_wrench(data)
    return vel

def pick_object(robot: URDriver.UniversalRobot) -> None:
    open_gripper()
    robot.control.moveL(PICK_P + Z_A*0.1)
    time.sleep(0.5)
    robot.control.moveL(PICK_P)
    time.sleep(0.5)
    close_gripper()
    robot.control.moveL(PICK_P + Z_A*0.1)
    time.sleep(0.5)

def place_object(robot: URDriver.UniversalRobot) -> None:
    close_gripper()
    robot.control.moveL(PLACE_P + Z_A*0.4)
    time.sleep(0.5)
    robot.control.moveL(PLACE_P + Z_A*0.27)
    # robot.control.moveL(PLACE_P + Z_A*0.14) #Rigid


def send_joint_states(q: np.ndarray):
    state = JointState()
    state.name = J_NAMES
    state.header = Header()
    state.header.stamp = rospy.Time.now()
    state.position = q
    js_publisher.publish(state)

def send_wrench(f: np.ndarray):
    wrench = WrenchStamped()
    wrench.header = Header()
    wrench.header.stamp     = rospy.Time.now()
    wrench.wrench.force.x   = f[0]
    wrench.wrench.force.y   = f[1]
    wrench.wrench.force.z   = f[2]
    wrench.wrench.torque.x  = f[3]
    wrench.wrench.torque.y  = f[4]
    wrench.wrench.torque.z  = f[5]
    wr_publisher.publish(wrench)

def open_gripper():
    control_gripper(id=0, operation_type=110, speed=0, position=0)

def close_gripper():
    control_gripper(id=0, operation_type=121, speed=30, position=0)

def free_gripper():
    control_gripper(id=0, operation_type=100, speed=0, position=0)




def main():

    global PLACE_P

    # Initialize median filter
    median = md.MedianFilter(NUM_JOINTS, 5)

    # Parameters
    velocity = 0.1
    acceleration = 0.1
    dt = 1.0/500  # 2ms
    lookahead_time = 0.1
    gain = 300

    # Initialize robot
    robot1 = URDriver.UniversalRobot('192.168.88.5')

    # Move to initial position
    robot1.control.moveJ(JOINTS)
    time.sleep(5)

    # Reset force sensor
    robot1.control.zeroFtSensor()
    robot1.update_state()

    circ_msg = rospy.wait_for_message('/circle_pose', Pose, timeout=None)
    print(circ_msg)

    circ_p = np.array([circ_msg.position.x, circ_msg.position.y, 0, 1])
    p_6, _ = robot_model.pose_angvec(robot1.state.q)
    r_6    = robot_model.rot(robot1.state.q)
    transform_6_0 = spm.SE3(*p_6) @ spm.SE3(spmb.r2t(r_6, check=True))
    transform_camera = transform_6_0 @ spm.SE3(-0.0305, -0.10218, 0.03445)
    circ_p = transform_camera.A.dot(circ_p)
    PLACE_P[0] = circ_p[0]
    PLACE_P[1] = circ_p[1]


    # Pick and place object
    # pick_object(robot1)
    place_object(robot1)

    # Reset force sensor
    robot1.control.zeroFtSensor()
    robot1.update_state()

    fdir = DIR
    z0 = 0
    z1 = 0
    dz = 0

    while True:
        start = time.time()

        # state
        robot1.update_state()
        send_joint_states(robot1.state.q)

        # Use medial filter to get forces
        fe = np.array(median.apply_median(robot1.state.f)).flatten()
        jac = robot_model.jacobian(robot1.state.q)

        # send_wrench(robot1.state.f)
        rot_6_0 = robot_model.rot(robot1.state.q)

        # Control
        if STAGE == 0:
            print('STAGE:', STAGE)
            fdir = find_object_force_control(fe, 20, rot_6_0)

        if STAGE == 1:
         #   print('STAGE:', STAGE)
            fdir = detect_hole_force_control(fe, 30)

        if STAGE == 2:
            print('STAGE:', STAGE)
            fdir = down_odirect_force_control(fe, 20, rot_6_0)



        # Calculate speed vector using Manipulator Jacobian
        jspeed = np.linalg.pinv(jac).dot(fdir)

        # Move manipulator
        robot1.control.speedJ(jspeed, acceleration, dt)

        end = time.time()
        duration = end - start
        if duration < dt:
            time.sleep(dt - duration)

        if rospy.is_shutdown():
            free_gripper()
            break


if __name__ == '__main__':

    current_filepath = os.path.dirname(os.path.abspath(__file__))
    urdf_filepath = os.path.join(current_filepath, '..', 'urdf_model', 'ur5e_fc.urdf')

    # ROS
    rospy.init_node('pick_and_place_node')
    robot_model = URDriver.RobotModel(urdf_filepath, 'base', TOOL)

    # Publishers
    js_publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
    wr_publisher = rospy.Publisher('/wrench', WrenchStamped, queue_size=10)

    # Services
    rospy.wait_for_service('/control_gripper')
    control_gripper = rospy.ServiceProxy('/control_gripper', control)

    main()