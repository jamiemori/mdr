# python native
import sys
import time
import socket
import queue
import threading
import math
import pdb

import pyaudio
import numpy as np

from random import randint

# config file
import config

# for lighting
import opc
import fastopc
# import board
# import busio
# import adafruit_mpr121

# audio processing
import dsp
import vis

from scipy.ndimage.filters import gaussian_filter1d

# pixels = [(0, 0, 0)] * config.NUM_LEDS
visualization_effect = config.VISUALIZATION_MODE
start = time.time()

######################################################################################
# audio analysis
######################################################################################

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
    output = np.concatenate((p[:, ::-1], p), axis=1)
    output = np.tile(output.transpose(), (config.NUM_STRIPS, 1))
    return output


def visualize_energy(y):
    """Effect that expands from the center with increasing sound energy"""
    global p

    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float((config.NUM_LEDS // 2) - 1)
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

    y = np.copy(interpolate(y, config.NUM_LEDS // 2))
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
    x = np.array([r, g, b]) * 255
    output = np.tile(x.transpose(), (config.NUM_STRIPS, 1))
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

    # Transform audio input into the frequency domain
    N = len(y_data)
    N_zeros = 2**int(np.ceil(np.log2(N))) - N

    # Pad with zeros until the next power of two
    y_data *= fft_window
    y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
    YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
    
    # Construct a Mel filterbank from the FFT data
    mel = np.atleast_2d(YS).T * dsp.mel_y.T
    # pdb.set_trace()

    # Scale data to values more suitable for visualization
    # mel = np.sum(mel, axis=0)
    mel = np.sum(mel, axis=0)
    mel = mel**2.0

    # Gain normalization
    mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
    mel /= mel_gain.value
    mel = mel_smoothing.update(mel)

    # map filterbank output onto led strip
    # pdb.set_trace()
    output = visualize_scroll(mel)
    vis.pixels = output
    vis.update()

def start_stream(callback):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=config.MIC_RATE,
                    input=True,
                    frames_per_buffer=samples_per_frame)
    overflows = 0
    prev_ovf_time = time.time()

    while True:
        try:
            y = np.frombuffer(stream.read(samples_per_frame, exception_on_overflow=False), 
                              dtype=np.int16)
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

# The previous time that the frames_per_second() function was called
_time_prev = time.time() * 1000.0

# The low-pass filter used to estimate frames-per-second
_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)

r_filt = dsp.ExpFilter(np.tile(0.01, config.NUM_LEDS // 2),
                       alpha_decay=0.2, alpha_rise=0.99)
g_filt = dsp.ExpFilter(np.tile(0.01, config.NUM_LEDS // 2),
                       alpha_decay=0.05, alpha_rise=0.3)
b_filt = dsp.ExpFilter(np.tile(0.01, config.NUM_LEDS // 2),
                       alpha_decay=0.1, alpha_rise=0.5)
common_mode = dsp.ExpFilter(np.tile(0.01, config.NUM_LEDS // 2),
                       alpha_decay=0.99, alpha_rise=0.01)
p_filt = dsp.ExpFilter(np.tile(1, (3, config.NUM_LEDS // 2)),
                       alpha_decay=0.1, alpha_rise=0.99)
p = np.tile(1.0, (3, config.NUM_LEDS // 2))
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
_prev_spectrum = np.tile(0.01, config.NUM_LEDS // 2)

# Number of audio samples to read every time frame
samples_per_frame = int(config.MIC_RATE / config.FPS)

# Array containing the rolling audio sample window
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16



######################################################################################
# audio analysis
######################################################################################

client = opc.Client("localhost:7890")
led_queue = queue.Queue()

class Rand:
    def __init__(self):
        self.last = None

    def __call__(self, max_num=7):
        r = randint(0, max_num)
        while r == self.last:
            r = random.randint(0, max_num)
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
    for j in range(config.NUM_LEDS):
        pixels[j] = (255, 255, 0)
        client.put_pixels(pixels)
        time.sleep(config.RATE)


def chase_1():
    while True:
        for j in range(config.NUM_LEDS):
            print(j)
            pixels = [(0, 0, 0)] * config.NUM_LEDS
            pixels[j] = (255, 0, 0)
            client.put_pixels(pixels)
            time.sleep(config.RATE)


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

    random_values = [random.random() for i in range(config.NUM_LEDS)]
    coordinates = list(
        range(led_coordinates[sensor_id][0], led_coordinates[sensor_id][1])
    )

    start_pixel = led_coordinates[sensor_id][0]
    end_pixel = led_coordinates[sensor_id][1]

    start_time = time.time()

    try:
        t_end = time.time() + 2.5
        while time.time() < t_end:
            t = time.time() - start_time
            pixels[start_pixel:end_pixel] = [
                pixel_color(t, i, config.NUM_LEDS, random_values) for i in coordinates
            ]
            client.put_pixels(pixels, channel=0)
    except KeyboardInterrupt:
        pixels = [(0, 0, 0)] * config.NUM_LEDS
        client.put_pixels(pixels, channel=0)
        sys.exit(1)
    return


def fade():
    """ fade effect """
    client = opc.Client("localhost:7890")

    black = [(0, 0, 0)] * config.NUM_LEDS
    white = [(255, 255, 255)] * config.NUM_LEDS

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

def main():
    vis.update()
    start_stream(microphone_update)

if __name__ == "__main__":
    main()
