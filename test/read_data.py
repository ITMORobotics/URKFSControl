from itertools import chain
import os,sys
import numpy as np
import pandas as pd
import logging
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time


robot_res = RTDEReceiveInterface('192.168.88.6')
robot_control = RTDEControlInterface('192.168.88.6')
robot_control.teachMode()

data = []
start_time = time.time()
while (time.time() - start_time) < 20.0:
    q = robot_res.getActualQ()
    pose = robot_res.getActualTCPPose()
    row = np.concatenate([np.array(q),np.array(pose)])
    print(row)
    data.append(row)
    time.sleep(0.1)

data_pd = pd.DataFrame(data, columns=['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'px', 'py', 'pz', 'rx', 'ry', 'rz'])
data_pd.to_csv("result.csv")
