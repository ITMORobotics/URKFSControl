from spatialmath import SE3,SO3, Twist3
import numpy as np
import roboticstoolbox
from spatialmath.base import animate,tranimate
import matplotlib.pyplot as plt


s1 = SE3(-0.15, -0.5, 0.35) @ SE3.Rx(np.pi, 'rad')
s2 = SE3(-0.15, -0.5, 0.5) @ SE3.Rx(np.pi-0.2, 'rad')

a = s1.interp(s2)
print(a)

trj = roboticstoolbox.tools.trajectory.ctraj(s1, s2, 11)

print(trj)
