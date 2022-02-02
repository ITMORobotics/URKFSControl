import numpy as np
import scipy
import sys, os
import PyKDL as kdl
from typing import Tuple
import time
from spatialmath import SE3,SO3, Twist3
from scipy.spatial.transform import Rotation as R

from gripper_controller import GripperSerialController

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf
from URDriver import robot
from coop import coop_robot
from URDriver import controller

class CoopSmartSystem():
    def __init__(self, ip1: str, ip2: str, left_urdf_filename: str, right_urdf_filename: str, tool_frame_name: str):

        self.__coop_P_matrix_full = scipy.linalg.block_diag(np.identity(3)*0.3, np.identity(3)*0.5, np.identity(3)*0.3, np.identity(3)*0.5)*18.0
        self.__coop_I_matrix_full = scipy.linalg.block_diag(np.identity(3)*0.02, np.identity(3)*0.01, np.identity(3)*0.02, np.identity(3)*0.01)*0.0008

        self.__coop_stiff_matrix_full = scipy.linalg.block_diag(np.identity(3)*0.005, np.identity(3)*0.05, np.identity(3)*0.005, np.identity(3)*0.05)

        self.__external_stiff_sM = coop_robot.generate_square_selection_matrix(np.array([0,0,0,0,0,0,1,1,1,1,1,1]))
        self.__internal_stiff_sM = coop_robot.generate_square_selection_matrix(np.array([1,1,1,1,1,1,0,1,1,1,1,1]))
        self.__full_control_sM = coop_robot.generate_square_selection_matrix(np.array([1,1,1,1,1,1,1,1,1,1,1,1]))
        print(self.__full_control_sM)

        self.__robot_model_left = robot.RobotModel(left_urdf_filename, 'world', tool_frame_name)
        self.__robot_model_right = robot.RobotModel(right_urdf_filename, 'world', tool_frame_name)
        self.__coop_model = coop_robot.DualCoopModel((self.__robot_model_left, self.__robot_model_right))
        self.__dt = 0.02
        self.__coop_ur = coop_robot.DualUniversalRobot(ip1, ip2, self.__dt)
        self.__grippers = {'left': GripperSerialController('/dev/gripper_left', 57600),'right': GripperSerialController('/dev/gripper_right', 57600)}
        time.sleep(1.5)
        self.__coop_hybride_controller = controller.CooperativeController(self.__coop_model, self.__coop_stiff_matrix_full, self.__coop_P_matrix_full, self.__coop_I_matrix_full, self.__dt)
        self.zeroFT()

    def zeroFT(self):
        # print(external_stiff_sM)
        ok = self.__coop_ur.control[0].zeroFtSensor()
        ok &= self.__coop_ur.control[1].zeroFtSensor()
        if not ok:
            raise(RuntimeError('Force torque connection was broken'))

    def get_state(self) -> coop_robot.CoopCartState:
        self.__coop_ur.update_state()
        return self.__coop_model.cart_state((self.__coop_ur.state[0].q, self.__coop_ur.state[1].q))

    def p2p_cartmove_avoid(self, to_state: coop_robot.CoopCartState, final_time: float, collide_check: bool):
        
        stiffnes_mode = False
        while True:
            self.__coop_ur.update_state()
            coop_state_from = self.__coop_model.cart_state((self.__coop_ur.state[0].q, self.__coop_ur.state[1].q))
            print("From tf:\n", coop_state_from.abs_tf)
            print("To tf:\n", to_state.abs_tf)
            trj = coop_robot.CoopSE3LineTrj(coop_state_from, to_state, self.__dt, final_time)

            start_time = time.time()

            target_coop_abs_pose1 = None 
            target_coop_abs_orient1 = None
            target_coop_rel_pose1 =None
            target_coop_rel_orient1 = None
            print("Continue executing")
            while time.time()-start_time < final_time + 2.0:
                start_loop_time = time.time()
                self.__coop_ur.update_state()
                coop_state_from = self.__coop_model.cart_state((self.__coop_ur.state[0].q, self.__coop_ur.state[1].q))
                trj_state_t = trj.getState(time.time()-start_time)

                target_coop_abs_pose1, target_coop_abs_orient1, target_coop_rel_pose1, target_coop_rel_orient1 = trj_state_t.to_pose_rot()
                print("Abs:\n", coop_state_from.abs_tf)
                print("Rel:\n", coop_state_from.rel_tf)
                control_dq = self.__coop_hybride_controller.hybride_world_control(
                    target_coop_abs_pose1,
                    target_coop_abs_orient1,
                    target_coop_rel_pose1,
                    target_coop_rel_orient1,

                    np.zeros(6),
                    np.zeros(6),
                    (self.__coop_ur.state[0].q, self.__coop_ur.state[1].q),
                    (self.__coop_ur.state[0].f, self.__coop_ur.state[1].f),
                    self.__full_control_sM[0], self.__full_control_sM[1]
                )

                # Send dq control to two robots
                # print(control_dq)
                self.__coop_ur.send_dq(control_dq)
                
                err = self.__coop_model.absolute_force_torque( (self.__coop_ur.state[0].f, self.__coop_ur.state[1].f))
                if np.linalg.norm(err) > 15.0 and collide_check:
                    print("Collision detected")
                    self.__coop_ur.stop()
                    stiffnes_mode = True
                    time.sleep(0.3)
                    break

                end_loop_time = time.time()
                duration = end_loop_time - start_loop_time
                if duration < self.__dt:
                    time.sleep(self.__dt - duration)
            self.__coop_hybride_controller.reset()
            if not stiffnes_mode:
                self.__coop_ur.stop()
                print("Executing finished")
                # time.sleep(1.0)
                break
            
            print("Start collision avoidance")
            self.stiff_control_mode(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), target_coop_rel_pose1, target_coop_rel_orient1, 3.0)
            stiffnes_mode = False
            self.__coop_hybride_controller.reset()
    
    def p2p_cart_handle_move(self, to_state: coop_robot.CoopCartState, final_time: float, collide_check: bool):
        stiffnes_mode = False
        handle_abs_force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        handle_rel_force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        while True:
            self.__coop_ur.update_state()
            coop_state_from = self.__coop_model.cart_state((self.__coop_ur.state[0].q, self.__coop_ur.state[1].q))
            print("From tf:\n", coop_state_from.abs_tf)
            print("To tf:\n", to_state.abs_tf)
            trj = coop_robot.CoopSE3LineTrj(coop_state_from, to_state, self.__dt, final_time)

            start_time = time.time()

            target_coop_abs_pose1 = None
            target_coop_abs_orient1 = None
            target_coop_rel_pose1 =None
            target_coop_rel_orient1 = None
            print("Continue executing")
            while time.time()-start_time < final_time + 2.0:
                start_loop_time = time.time()
                self.__coop_ur.update_state()
                coop_state_from = self.__coop_model.cart_state((self.__coop_ur.state[0].q, self.__coop_ur.state[1].q))
                trj_state_t = trj.getState(time.time()-start_time)

                target_coop_abs_pose1, target_coop_abs_orient1, target_coop_rel_pose1, target_coop_rel_orient1 = trj_state_t.to_pose_rot()
                print(trj_state_t.abs_tf)
                control_dq = self.__coop_hybride_controller.hybride_world_control(
                    target_coop_abs_pose1,
                    target_coop_abs_orient1,
                    target_coop_rel_pose1,
                    target_coop_rel_orient1,

                    handle_abs_force,
                    handle_rel_force,
                    (self.__coop_ur.state[0].q, self.__coop_ur.state[1].q),
                    (self.__coop_ur.state[0].f, self.__coop_ur.state[1].f),
                    self.__internal_stiff_sM[0], self.__internal_stiff_sM[1]
                )

                # Send dq control to two robots
                # print(control_dq)
                self.__coop_ur.send_dq(control_dq)
                
                err = self.__coop_model.absolute_force_torque( (self.__coop_ur.state[0].f, self.__coop_ur.state[1].f))
                if np.linalg.norm(err) > 15.0 and collide_check:
                    print("Collision detected")
                    self.__coop_ur.stop()
                    stiffnes_mode = True
                    time.sleep(0.3)
                    break

                end_loop_time = time.time()
                duration = end_loop_time - start_loop_time
                if duration < self.__dt:
                    time.sleep(self.__dt - duration)
            self.__coop_hybride_controller.reset()
            if not stiffnes_mode:
                self.__coop_ur.stop()
                print("Executing finished")
                # time.sleep(1.0)
                break
            
            print("Start collision avoidance")
            self.__coop_ur.update_state()
            coop_state_now = self.__coop_model.cart_state((self.__coop_ur.state[0].q, self.__coop_ur.state[1].q))
            target_coop_abs_pose1, target_coop_abs_orient1, target_coop_rel_pose1, target_coop_rel_orient1 = coop_state_now.to_pose_rot()
            self.stiff_control_mode(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), target_coop_rel_pose1, target_coop_rel_orient1, 3.0)
            # time.sleep(5.0)
            stiffnes_mode = False
            self.__coop_hybride_controller.reset()
    
    def stiff_control_mode(self,
        target_coop_abs_ft: np.ndarray,
        target_coop_rel_pose: np.ndarray,
        target_coop_rel_orient: np.ndarray,
        sec_delay: float = 30.0
        ):
        time_start_free = time.time()
        while True:
            self.__coop_ur.update_state()
            err = self.__coop_model.absolute_force_torque( (self.__coop_ur.state[0].f, self.__coop_ur.state[1].f))
            norm_err = np.linalg.norm(err)
            print(norm_err)
            if norm_err > 10.0:
                time_start_free = time.time()

            control_dq = self.__coop_hybride_controller.hybride_world_control(
                np.zeros(3),
                np.identity(3),
                target_coop_rel_pose,
                target_coop_rel_orient,

                target_coop_abs_ft,
                np.zeros(6),
                (self.__coop_ur.state[0].q, self.__coop_ur.state[1].q),
                (self.__coop_ur.state[0].f, self.__coop_ur.state[1].f),
                self.__external_stiff_sM[0], self.__external_stiff_sM[1]
            )
            
            self.__coop_ur.send_dq(control_dq)
            if time.time()-time_start_free > sec_delay:
                break

    def open_gripper(self, gripper_names: Tuple[str]):
        for g in gripper_names:
            self.__grippers[g].open()
            time.sleep(0.1)

    def close_gripper(self, gripper_names: Tuple[str]):
        for g in gripper_names:
            self.__grippers[g].close_torque(15)
            time.sleep(0.1)