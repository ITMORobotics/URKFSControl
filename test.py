import URDriver
import time

from URDriver import robot

robot1 = URDriver.UniversalRobot('192.168.88.5')
robot2 = URDriver.UniversalRobot('192.168.88.6')
robot1.control.teachMode()
robot2.control.teachMode()

while True:
    robot1.update_state()
    robot2.update_state()
    print('r1', robot1.state.q)
    print('r2', robot2.state.q)
    time.sleep(0.002)
