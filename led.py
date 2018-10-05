import sys
# import opc
import time
# import board
# import busio
import socket

from random import randint

import adafruit_mpr121

NUM_LEDS = 512

def send():
    HOST = "192.168.1.224"
    PORT = 5555 
    MESSAGE = "Hello, World!"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall((5).to_bytes(2, byteorder='little'))


# def fade():
    # """ fade effect """
    # client = opc.Client("localhost:7890")

    # black = [(0, 0, 0)] * NUM_LEDS
    # white = [(255, 255, 255)] * NUM_LEDS

    # while True:
        # client.put_pixels(white)
        # time.sleep(0.05)

        # client.put_pixels(black)
        # time.sleep(0.05)


# def strobe():
    # """ strobe effect """
    # num_leds = 512
    # client = opc.Client("localhost:7890")

    # black = [(0, 0, 0)] * num_leds
    # white = [(255, 255, 255)] * num_leds

    # while True:
        # client.put_pixels(white)
        # client.put_pixels(black)
        # time.sleep(0.05)


class Rand:
    def __init__(self):
        self.last = None

    def __call__(self):
        r = randint(0, 11)
        while r == self.last:
            r = random.randint(0, 11)
        self.last = r
        return r


# def game_mode():
    # i2c = busio.I2C(board.SCL, board.SDA)
    # mpr121 = adafruit_mpr121.MPR121(i2c)

    # # # NOTE you can optionally change the address of the device:
    # # mpr121 = adafruit_mpr121.MPR121(i2c, address=0x91)

    # # initial touch state
    # last_touched = mpr121.touched()

    # # generate random number for touch tensor
    # # mpr121 sensor has 12 sensors total, so need 0-11
    # to_be_touched = Rand()

    # # Loop forever testing each input and printing when they're touched.
    # while True:
        # current_touched = mpr121.touched()

        # for i in range(12):
            # # Each pin is represented by a bit in the touched value.
            # # A value of 1 means the pin is being touched, and 0 means
            # # it is not being touched.
            # pin_bit = 1 << i

            # if (
                # to_be_touched == i
                # and current_touched & pin_bit
                # and not last_touched & pin_bit
            # ):
                # print("{0} touched!".format(i))
                # to_be_touched = Rand()

            # if (
                # to_be_touched == i
                # and not current_touched & pin_bit
                # and last_touched & pin_bit
            # ):
                # print("{0} released!".format(i))

        # # Update last state and wait a short period before repeating.
        # last_touched = current_touched
        # time.sleep(0.1)

        # # for debugging
        # print("\t\t\t\t\t\t\t\t\t\t\t\t\t 0x{0:0X}".format(cap.touched()))
        # filtered = [cap.filtered_data(i) for i in range(12)]

        # print("Filt:", "\t".join(map(str, filtered)))
        # base = [cap.baseline_data(i) for i in range(12)]

        # print("Base:", "\t".join(map(str, base)))


def main():
    send()
    # # TODO add option selection based on input mode
    # game_mode()

if __name__ == "__main__":
    main()
