import serial
import sys
import os

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
    serial_ports = search_serial_ports()
    grippers_dict = {}

    for counter, value in enumerate(serial_ports):
        grippers_dict[counter] = GripperSerialController(value, 57600)

    print(girppers_dict)
    print("Prekol")


if __name__ == "__main__":
    main()

