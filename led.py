import sys
import opc
import time
import socket
import math
import queue

import board
import busio
import threading

import random
from random import randint

import adafruit_mpr121


NUM_LEDS = 792
#NUM_LEDS = 512
client = opc.Client("localhost:7890")
k = 0.001
pixels = [(0, 0, 0)] * NUM_LEDS


def send(pin):
    HOST = "192.168.1.224"
    PORT = 5555

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall((pin).to_bytes(2, byteorder="little"))


def chase_0():
    while True:
        pixels = [(0, 0, 0)] * NUM_LEDS
        for j in range(NUM_LEDS):
            print(j)
            pixels[j] = (255, 255, 0)
            client.put_pixels(pixels)
            time.sleep(k)

def chase_1():
    while True:
        for j in range(NUM_LEDS):
            print(j)
            pixels = [(0, 0, 0)] * NUM_LEDS
            pixels[j] = (255, 0, 0)
            client.put_pixels(pixels)
            time.sleep(k)


def remap(x, oldmin, oldmax, newmin, newmax):
    """
    Remap the float x from the range oldmin-oldmax to the range newmin-newmax

    Does not clamp values that exceed min or max.
    For example, to make a sine wave that goes between 0 and 256:
        remap(math.sin(time.time()), -1, 1, 0, 256)
    """
    zero_to_one = (x - oldmin) / (oldmax - oldmin)
    return zero_to_one * (newmax - newmin) + newmin


def clamp(x, minn, maxx):
    """Restrict the float x to the range minn-maxx."""
    return max(minn, min(maxx, x))


def cos(x, offset=0, period=1, minn=0, maxx=1):
    """A cosine curve scaled to fit in a 0-1 range and 0-1 domain by default.
    offset: how much to slide the curve across the domain (should be 0-1)
    period: the length of one wave
    minn, maxx: the output range
    """
    value = math.cos((x / period - offset) * math.pi * 2) / 2 + 0.5
    return value * (maxx - minn) + minn


def contrast(color, center, mult):
    """
    Expand the color values by a factor of mult around the 
    pivot value of center.

    color: an (r, g, b) tuple
    center: a float -- the fixed point
    mult: a float -- expand or contract the values around the center point
    """
    r, g, b = color
    r = (r - center) * mult + center
    g = (g - center) * mult + center
    b = (b - center) * mult + center
    return (r, g, b)


def clip_black_by_luminance(color, threshold):
    """If the color's luminance is less than threshold, replace it with black.
    
    color: an (r, g, b) tuple
    threshold: a float
    """
    r, g, b = color
    if r + g + b < threshold * 3:
        return (0, 0, 0)
    return (r, g, b)


def clip_black_by_channels(color, threshold):
    """Replace any individual r, g, or b value less than threshold with 0.
    color: an (r, g, b) tuple
    threshold: a float
    """
    r, g, b = color
    if r < threshold:
        r = 0
    if g < threshold:
        g = 0
    if b < threshold:
        b = 0
    return (r, g, b)


def mod_dist(a, b, n):
    """Return the distance between floats a and b, modulo n.
    The result is always non-negative.
    For example, thinking of a clock:
    mod_dist(11, 1, 12) == 2 because you can "wrap around".
    """
    return min((a - b) % n, (b - a) % n)


def gamma(color, gamma):
    """
    Apply a gamma curve to the color.  
    The color values should be in the range 0-1.
    """
    r, g, b = color
    return (max(r, 0) ** gamma, max(g, 0) ** gamma, max(b, 0) ** gamma)


def pixel_color(t, i, n_pixels, random_values):
    """
    from: 
    https://github.com/zestyping/openpixelcontrol/blob/master/python/miami.py

    Compute the color of a given pixel.

    t: time in seconds since the program started.
    i: which pixel this is, starting at 0
    n_pixels: the total number of pixels
    random_values: a list containing a constant random value for each pixel

    Returns an (r, g, b) tuple in the range 0-255
    """
    y = cos(i + 0.2 * i, offset=0, period=1, minn=0, maxx=0.6)

    # make x, y, z -> r, g, b sine waves
    r = cos(y, offset=t / 4, period=2.5, minn=0, maxx=1)
    g = cos(y, offset=t / 4, period=2.5, minn=0, maxx=1)
    b = cos(y, offset=t / 4, period=2.5, minn=0, maxx=1)

    r, g, b = contrast((r, g, b), 0.5, 1.4)

    clampdown = (r + g + b) / 2
    clampdown = remap(clampdown, 0.4, 0.5, 0, 1)
    clampdown = clamp(clampdown, 0, 1)
    clampdown *= 0.9

    r *= clampdown
    g *= clampdown
    b *= clampdown

    # black out regions
    r2 = cos(i, offset=t / 10 + 12.345, period=4, minn=0, maxx=1)
    g2 = cos(i, offset=t / 10 + 24.536, period=4, minn=0, maxx=1)
    b2 = cos(i, offset=t / 10 + 34.675, period=4, minn=0, maxx=1)
    clampdown = (r2 + g2 + b2) / 2
    clampdown = remap(clampdown, 0.2, 0.3, 0, 1)
    clampdown = clamp(clampdown, 0, 1)
    r *= clampdown
    g *= clampdown
    b *= clampdown

    # color scheme: fade towards blue-and-orange
    g = g * 0.6 + ((r + b) / 2) * 0.4

    # fade behind twinkle
    fade = cos(t - i / n_pixels, offset=0, period=7, minn=0, maxx=1) ** 20
    fade = 1 - fade * 0.2
    r *= fade
    g *= fade
    b *= fade

    # twinkle occasional LEDs
    twinkle_speed = 0.7
    twinkle_density = 0.4
    twinkle = (random_values[i] * 10 + time.time() * twinkle_speed) % 1
    twinkle = abs(twinkle * 2 - 1)
    twinkle = remap(twinkle, 0, 1, -1 / twinkle_density, 1.1)
    twinkle = clamp(twinkle, -0.5, 1.1)
    twinkle **= 5
    twinkle *= cos(t - i / n_pixels, offset=0, period=7, minn=0, maxx=1) ** 20
    twinkle = clamp(twinkle, -0.3, 1)
    r += twinkle
    g += twinkle
    b += twinkle

    # apply gamma curve, only do this on live leds, not in the simulator
    r, g, b = gamma((r, g, b), 1)

    return (r * 256, g * 256, b * 128)


def miami(sensor_id):
    """ 
    based on:
    https://github.com/zestyping/openpixelcontrol/blob/master/python/miami.py 
    """
    led_coordinates = {
        0: (0, 64),
        1: (65, 128),
        2: (129, 192),
        3: (193, 256), 
        4: (257, 320), 
        5: (321, 385), 
        6: (386, 449), 
        7: (450, 512)
    }

    global pixels

    random_values = [random.random() for i in range(NUM_LEDS)]
    coordinates = list(
        range(led_coordinates[sensor_id][0], led_coordinates[sensor_id][1])
    )

    start_pixel = led_coordinates[sensor_id][0]
    end_pixel = led_coordinates[sensor_id][1]

    print(start_pixel, end_pixel)
    print(coordinates)

    start_time = time.time()

    try:
        t_end = time.time() + 2.5
        while time.time() < t_end:
            t = time.time() - start_time
            pixels[start_pixel:end_pixel] = [pixel_color(t, i, NUM_LEDS, random_values) for i in coordinates]
            client.put_pixels(pixels, channel=0)
    except KeyboardInterrupt:
        pixels = [(0, 0, 0)] * NUM_LEDS
        client.put_pixels(pixels, channel=0)
        sys.exit(1)
    return


def fade():
    """ fade effect """
    client = opc.Client("localhost:7890")

    black = [(0, 0, 0)] * NUM_LEDS
    white = [(255, 255, 255)] * NUM_LEDS

    while True:
        client.put_pixels(white)
        time.sleep(0.05)

        client.put_pixels(black)
        time.sleep(0.05)


def strobe():
    """ strobe effect """
    num_leds = 512
    client = opc.Client("localhost:7890")

    black = [(0, 0, 0)] * num_leds
    white = [(255, 255, 255)] * num_leds

    while True:
        client.put_pixels(white)
        client.put_pixels(black)
        time.sleep(0.05)


class Rand:
    def __init__(self):
        self.last = None

    def __call__(self):
        r = randint(0, 7)
        while r == self.last:
            r = random.randint(0, 7)
        self.last = r
        return r


led_queue = queue.Queue()


def execute_lights(led_queue):
    """ 
    lights up light strip based on sensor_id

    uses threading for non-blocking
    """
    while True:
        sensor_id = led_queue.get()
        print(sensor_id)
        miami(sensor_id)
        led_queue.task_done()


def explore_test():
    worker_threads = []
    for i in range(5):
        t = threading.Thread(target=execute_lights, daemon=True, args=(led_queue,))
        worker_threads.append(t)
        t.start()

    while True:
        for i in range(5):
            time.sleep(0.5)
            print(i)
            rand = Rand()
            led_queue.put(rand())

        j = 2
        for j in range(3): 
            print('sleeping', j)
            time.sleep(1)
            j -= 1


def explore_mode():
    i2c = busio.I2C(board.SCL, board.SDA)
    mpr121 = adafruit_mpr121.MPR121(i2c)

    worker_threads = []
    for i in range(20):
        t = threading.Thread(target=execute_lights, daemon=True, args=(led_queue,))
        worker_threads.append(t)
        t.start()

    last_touched = mpr121.touched()
    while True:
        current_touched = mpr121.touched()

        # Check each pin's last and current state to see if it was pressed or released.
        for i in range(12):
            # Each pin is represented by a bit in the touched value.  A value of 1
            # means the pin is being touched, and 0 means it is not being touched.
            pin_bit = 1 << i
            # First check if transitioned from not touched to touched.
            if current_touched & pin_bit and not last_touched & pin_bit:
                print("{0} touched!".format(i))
                if i <= 7:
                    led_queue.put(i)
                # send(i)

            if not current_touched & pin_bit and last_touched & pin_bit:
                print("{0} released!".format(i))

        # Update last state and wait a short period before repeating.
        last_touched = current_touched
        time.sleep(0.1)

    # # # NOTE you can optionally change the address of the device:
    # # mpr121 = adafruit_mpr121.MPR121(i2c, address=0x91)

    # # initial touch state
    # last_touched = mpr121.touched()

    # while True:
    # current_touched = mpr121.touched()
    # for i in range(12):
    # # Each pin is represented by a bit in the touched value.
    # # A value of 1 means the pin is being touched, and 0 means
    # # it is not being touched.
    # pin_bit = 1 << i

    # if (
    # current_touched & pin_bit
    # and not last_touched & pin_bit
    # ):
    # print("{0} touched!".format(i))
    # send(i)

    # if (
    # not current_touched & pin_bit
    # and last_touched & pin_bit
    # ):
    # print("{0} released!".format(i))

    # # Update last state and wait a short period before repeating.
    # last_touched = current_touched
    # time.sleep(0.1)

    # # for debugging
    # print("\t\t\t\t\t\t\t\t\t\t\t\t\t 0x{0:0X}".format(mpr121.touched()))
    # filtered = [mpr121.filtered_data(i) for i in range(12)]

    # print("Filt:", "\t".join(map(str, filtered)))
    # base = [mpr121.baseline_data(i) for i in range(12)]

    # print("Base:", "\t".join(map(str, base)))


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


def main():
    #explore_mode()
    #explore_test()
    #while True:
        #miami()
    #chase_0()
    chase_1()
    # send()

    # # TODO add option selection based on input mode
    # game_mode()

if __name__ == "__main__":
    main()
