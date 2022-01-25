import serial
import sys
import os
import glob
import logging
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gripper_controller import GripperSerialController

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


def main():
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger("Grippers")
    serial_ports = search_serial_ports()
    grippers_dict = {}

    for counter, value in enumerate(serial_ports):
        if 'ACM' in value:
            grippers_dict[counter] = GripperSerialController(value, 57600)
            time.sleep(0.2)
    
    logger.info(" Found %d grippers"%len(grippers_dict))
    for i in range(20):    
        grippers_dict[0].close()
        grippers_dict[1].close()
        time.sleep(3)
        grippers_dict[0].open()
        grippers_dict[1].open()
        time.sleep(3)

    #for counter, _ in enumerate(grippers_dict):
        #grippers_dict[counter].close()

if __name__ == "__main__":
    main()

