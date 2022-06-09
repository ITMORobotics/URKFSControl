from itertools import chain
import os,sys
import numpy as np
import pandas as pd
import logging
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time


robot_res = RTDEReceiveInterface('192.168.88.5')
robot_control = RTDEControlInterface('192.168.88.5')
robot_control.teachMode()

robot_res2 = RTDEReceiveInterface('192.168.88.6')
robot_control2 = RTDEControlInterface('192.168.88.6')
robot_control2.teachMode()

dt=0.02
data = []
start_time = time.time()

while (time.time() - start_time) < 100.0:
    start_loop_time = time.time()
    q = robot_res.getActualQ()
    pose = robot_res.getActualTCPPose()
    q2 = robot_res2.getActualQ()
    pose2 = robot_res2.getActualTCPPose()
    row = np.concatenate([np.array(q),np.array(pose), np.array(q2),np.array(pose2)])
    print(row)
    data.append(row)

    end_loop_time = time.time()
    duration = end_loop_time - start_loop_time
    if duration < dt:
        time.sleep(dt - duration)

data_pd = pd.DataFrame(data, columns=['ql0', 'ql1', 'ql2', 'ql3', 'ql4', 'ql5', 'plx', 'ply', 'plz', 'rlx', 'rly', 'rlz', 'qr0', 'qr1', 'qr2', 'qr3', 'qr4', 'qr5', 'prx', 'pry', 'prz', 'rrx', 'rry', 'rrz'])
data_pd.to_csv("result2.csv")
