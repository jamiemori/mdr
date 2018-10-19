# python native
import sys
import time
import socket
import queue
import threading

# import Rand class
from utils import *

# all LED effects
from effects import *

import board
import busio

import adafruit_mpr121

import opc
import math

from random import randint

import time
import numpy as np
import pyaudio
import config
import numpy as np

from scipy.ndimage.filters import gaussian_filter1d
import config

import dsp
import vis

NUM_LEDS = 792
pixels = [(0, 0, 0)] * NUM_LEDS
rate = 0.001

_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""

_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)
"""The low-pass filter used to estimate frames-per-second"""

r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.2, alpha_rise=0.99)
g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.05, alpha_rise=0.3)
b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.1, alpha_rise=0.5)
common_mode = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.99, alpha_rise=0.01)
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)),
                       alpha_decay=0.1, alpha_rise=0.99)
p = np.tile(1.0, (3, config.N_PIXELS // 2))
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)

fft_plot_filter = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.5, alpha_rise=0.99)
mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD,
                       alpha_decay=0.02, alpha_rise=0.02)
fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)

prev_fps_update = time.time()
_prev_spectrum = np.tile(0.01, config.N_PIXELS // 2)

# Number of audio samples to read every time frame
samples_per_frame = int(config.MIC_RATE / config.FPS)

# Array containing the rolling audio sample window
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16

visualization_effect = visualize_spectrum
"""Visualization effect to display on the LED strip"""


def frames_per_second():
    """Return the estimated frames per second
    Returns the current estimate for frames-per-second (FPS).
    FPS is estimated by measured the amount of time that has elapsed since
    this function was previously called. The FPS estimate is low-pass filtered
    to reduce noise.
    This function is intended to be called one time for every iteration of
    the program's main loop.
    Returns
    -------
    fps : float
        Estimated frames-per-second. This value is low-pass filtered
        to reduce noise.
    """
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)


def memoize(function):
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper

@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)


def interpolate(y, new_length):
    """Intelligently resizes the array by linearly interpolating the values
    Parameters
    ----------
    y : np.array
        Array that should be resized
    new_length : int
        The length of the new interpolated array
    Returns
    -------
    z : np.array
        New array with length of new_length that contains the interpolated
        values of y.
    """
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z


def visualize_scroll(y):
    """Effect that originates in the center and scrolls outwards"""
    global p
    y = y**2.0
    gain.update(y)
    y /= gain.value
    y *= 255.0
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))

    # Scrolling effect window
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    p = gaussian_filter1d(p, sigma=0.2)

    # Create new color originating at the center
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b

    # Update the LED strip
    return np.concatenate((p[:, ::-1], p), axis=1)


def visualize_energy(y):
    """Effect that expands from the center with increasing sound energy"""
    global p
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.9
    r = int(np.mean(y[:len(y) // 3]**scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = int(np.mean(y[2 * len(y) // 3:]**scale))
    # Assign color to different frequency regions
    p[0, :r] = 255.0
    p[0, r:] = 0.0
    p[1, :g] = 255.0
    p[1, g:] = 0.0
    p[2, :b] = 255.0
    p[2, b:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=4.0)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=4.0)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=4.0)
    # Set the new pixel value
    return np.concatenate((p[:, ::-1], p), axis=1)


def visualize_spectrum(y):
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""

    global _prev_spectrum
    y = np.copy(interpolate(y, config.N_PIXELS // 2))
    common_mode.update(y)
    diff = y - _prev_spectrum
    _prev_spectrum = np.copy(y)

    # Color channel mappings
    r = r_filt.update(y - common_mode.value)
    g = np.abs(diff)
    b = b_filt.update(np.copy(y))

    # Mirror the color channels for symmetric output
    r = np.concatenate((r[::-1], r))
    g = np.concatenate((g[::-1], g))
    b = np.concatenate((b[::-1], b))
    output = np.array([r, g,b]) * 255
    return output

def microphone_update(audio_samples):

    global y_roll, prev_rms, prev_exp, prev_fps_update

    # Normalize samples between 0 and 1
    y = audio_samples / 2.0**15

    # Construct a rolling window of audio samples
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)
    vol = np.max(np.abs(y_data))

    if vol < config.MIN_VOLUME_THRESHOLD:
        print('No audio input. Volume below threshold. Volume:', vol)
        vis.pixels = np.tile(0, (3, config.N_PIXELS))
        vis.update()
    else:
        # Transform audio input into the frequency domain
        N = len(y_data)
        N_zeros = 2**int(np.ceil(np.log2(N))) - N

        # Pad with zeros until the next power of two
        y_data *= fft_window
        y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
        YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
        
        # Construct a Mel filterbank from the FFT data
        mel = np.atleast_2d(YS).T * dsp.mel_y.T

        # Scale data to values more suitable for visualization
        # mel = np.sum(mel, axis=0)
        mel = np.sum(mel, axis=0)
        mel = mel**2.0

        # Gain normalization
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= mel_gain.value
        mel = mel_smoothing.update(mel)

        # Map filterbank output onto LED strip
        output = visualization_effect(mel)
        vis.pixels = output
        vis.update()
    
def start_stream(callback):
    p = pyaudio.PyAudio()
    frames_per_buffer = int(config.MIC_RATE / config.FPS)
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=config.MIC_RATE,
                    input=True,
                    frames_per_buffer=frames_per_buffer)
    overflows = 0
    prev_ovf_time = time.time()

    while True:
        try:
            y = np.fromstring(stream.read(frames_per_buffer), dtype=np.int16)
            y = y.astype(np.float32)
            callback(y)
        except IOError:
            overflows += 1
            if time.time() > prev_ovf_time + 1:
                prev_ovf_time = time.time()
                print('Audio buffer has overflowed {} times'.format(overflows))
    stream.stop_stream()
    stream.close()
    p.terminate()


######################################################################################
# audio spectrum
######################################################################################


client = opc.Client("localhost:7890")
led_queue = queue.Queue()

class Rand:
    def __init__(self):
        self.last = None

    def __call__(self):
        r = randint(0, 7)
        while r == self.last:
            r = random.randint(0, 7)
        self.last = r
        return r


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


def chase_0():
    while True:
        pixels = [(0, 0, 0)] * NUM_LEDS
        for j in range(NUM_LEDS):
            print(j)
            pixels[j] = (255, 255, 0)
            client.put_pixels(pixels)
            time.sleep(rate)


def chase_1():
    while True:
        for j in range(NUM_LEDS):
            print(j)
            pixels = [(0, 0, 0)] * NUM_LEDS
            pixels[j] = (255, 0, 0)
            client.put_pixels(pixels)
            time.sleep(rate)


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

    return (r * 256, g * 128, b * 128)


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
        7: (450, 512),
    }

    global pixels

    random_values = [random.random() for i in range(NUM_LEDS)]
    coordinates = list(
        range(led_coordinates[sensor_id][0], led_coordinates[sensor_id][1])
    )

    start_pixel = led_coordinates[sensor_id][0]
    end_pixel = led_coordinates[sensor_id][1]

    #print(start_pixel, end_pixel)
    #print(coordinates)

    start_time = time.time()

    try:
        t_end = time.time() + 2.5
        while time.time() < t_end:
            t = time.time() - start_time
            pixels[start_pixel:end_pixel] = [
                pixel_color(t, i, NUM_LEDS, random_values) for i in coordinates
            ]
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



def send(pin):
    HOST = "192.168.1.224"
    PORT = 5555

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall((pin).to_bytes(2, byteorder="little"))


def execute_lights(led_queue):
    """ 
    lights up light strip based on sensor_id

    uses threading for non-blocking
    """
    while True:
        sensor_id = led_queue.get()
        miami(sensor_id)
        led_queue.task_done()


def explore_test():
    worker_threads = []
    for i in range(5):
        t = threading.Thread(
            target=execute_lights, daemon=True, args=(led_queue,)
        )
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
        print("sleeping", j)
        time.sleep(1)
        j -= 1


def explore_mode(debug=False):
    i2c = busio.I2C(board.SCL, board.SDA)
    mpr121 = adafruit_mpr121.MPR121(i2c)

    worker_threads = []
    for i in range(20):
        t = threading.Thread(
            target=execute_lights, daemon=True, args=(led_queue,)
        )
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
                    send(i)

            if not current_touched & pin_bit and last_touched & pin_bit:
                print("{0} released!".format(i))

        # Update last state and wait a short period before repeating.
        last_touched = current_touched
        time.sleep(0.1)

        if debug:
            # for debugging
            print(
                "\t\t\t\t\t\t\t\t\t\t\t\t\t 0x{0:0X}".format(mpr121.touched())
            )
            filtered = [mpr121.filtered_data(i) for i in range(12)]

            print("Filt:", "\t".join(map(str, filtered)))
            base = [mpr121.baseline_data(i) for i in range(12)]

            print("Base:", "\t".join(map(str, base)))


def game_mode(debug):
    i2c = busio.I2C(board.SCL, board.SDA)
    mpr121 = adafruit_mpr121.MPR121(i2c)

    # # NOTE you can optionally change the address of the device:
    # mpr121 = adafruit_mpr121.MPR121(i2c, address=0x91)

    # initial touch state
    last_touched = mpr121.touched()

    # generate random number for touch tensor
    # mpr121 sensor has 12 sensors total, so need 0-11
    to_be_touched = Rand()

    # Loop forever testing each input and printing when they're touched.
    while True:
        current_touched = mpr121.touched()

        for i in range(12):
            # Each pin is represented by a bit in the touched value.
            # A value of 1 means the pin is being touched, and 0 means
            # it is not being touched.
            pin_bit = 1 << i

            if (
                to_be_touched == i
                and current_touched & pin_bit
                and not last_touched & pin_bit
            ):
                print("{0} touched!".format(i))
                to_be_touched = Rand()

            if (
                to_be_touched == i
                and not current_touched & pin_bit
                and last_touched & pin_bit
            ):
                print("{0} released!".format(i))

        # Update last state and wait a short period before repeating.
        last_touched = current_touched
        time.sleep(0.1)

        if debug:
            # for debugging
            print(
                "\t\t\t\t\t\t\t\t\t\t\t\t\t 0x{0:0X}".format(mpr121.touched())
            )
            filtered = [mpr121.filtered_data(i) for i in range(12)]

            print("Filt:", "\t".join(map(str, filtered)))
            base = [mpr121.baseline_data(i) for i in range(12)]

            print("Base:", "\t".join(map(str, base)))


def main():
    # Initialize LEDs
    vis.update()

    # Start listening to live audio stream
    start_stream(microphone_update)

    # # TODO add option selection based on input mode
    # explore_mode()
    # game_mode()

if __name__ == "__main__":
    main()
