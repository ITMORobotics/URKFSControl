import serial
import sys
import os
import glob
import logging
import time
from math import pi
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gripper_controller import GripperSerialController
import URDriver

def search_serial_ports():
        if sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/tty[A-Za-z]*')
        else:
            raise EnvironmentError('Unsupported platform')
        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except(OSError, serial.SerialException):
                pass
        return result

def test():
    serial_ports = search_serial_ports()
    grippers_dict = {}
    for i in range(20):
        for counter, value in enumerate(serial_ports):
             if 'ACM' in value:
                 grippers_dict[counter] = GripperSerialController(value, 57600)
                 time.sleep(1)
        grippers_dict[0].open()
        grippers_dict[1].open()
        time.sleep(3)
        grippers_dict[0].close()
        grippers_dict[1].close()
        del grippers_dict[0]
        del grippers_dict[1]
        time.sleep(2)


def main():
    robot5 = URDriver.UniversalRobot('192.168.88.5')
    robot6 = URDriver.UniversalRobot('192.168.88.6')
    
    final_val5 = np.array([85.67*pi/180, -88.73*pi/180, -127.58*pi/180, -48.65*pi/180, 105.26*pi/180, -4.05*pi/180])
    final_val6 = np.array([79.53*pi/180, -105.9*pi/180, 62.09*pi/180, -55.91*pi/180, -95.51*pi/180, 115.83*pi/180])

    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger("Grippers")
    serial_ports = search_serial_ports()
    grippers_dict = {}

    for counter, value in enumerate(serial_ports):
        if 'ACM' in value:
            grippers_dict[counter] = GripperSerialController(value, 57600)
            time.sleep(1)
    
    logger.info(" Found %d grippers"%len(grippers_dict))
    
    grippers_dict[0].open()
    grippers_dict[1].open()
    time.sleep(3)
    robot5.control.moveJ(final_val5)
    robot6.control.moveJ(final_val6)
    
    grippers_dict[0].close()
    grippers_dict[1].close()


    #for i in range(20):    
        #grippers_dict[0].close()
        #grippers_dict[1].close()
        #time.sleep(3)
        #grippers_dict[0].open()
        #grippers_dict[1].open()
        #time.sleep(3)

    #for counter, _ in enumerate(grippers_dict):
        #grippers_dict[counter].close()

if __name__ == "__main__":
    test()

