import os

DEVICE = 'pi'
NUM_LEDS = 64
NUM_STRIPS = 8
RATE = 0.001

"""GPIO pin connected to the LED strip pixels (must support PWM)"""
LED_PIN = 18

"""LED signal frequency in Hz (usually 800kHz)"""
LED_FREQ_HZ = 800000

"""DMA channel used for generating PWM signal (try 5)"""
LED_DMA = 5

"""Brightness of LED strip between 0 and 255"""
BRIGHTNESS = 255

"""Set True if using an inverting logic level converter"""
LED_INVERT = True

"""Set to True because Raspberry Pi doesn't use hardware dithering"""
SOFTWARE_GAMMA_CORRECTION = False

"""Whether to display the FPS when running (can reduce performance)"""
DISPLAY_FPS = False

"""Sampling frequency of the microphone in Hz"""
MIC_RATE = 44100

"""Visualization mode"""
VISUALIZATION_MODE = 2

"""
Desired refresh rate of the visualization (frames per second)

FPS indicates the desired refresh rate, or frames-per-second, of the audio
visualization. The actual refresh rate may be lower if the computer cannot keep
up with desired FPS value.

Higher framerates improve "responsiveness" and reduce the latency of the
visualization but are more computationally expensive.

Low framerates are less computationally expensive, but the visualization may
appear "sluggish" or out of sync with the audio being played if it is too low.

The FPS should not exceed the maximum refresh rate of the LED strip, which
depends on how long the LED strip is.
"""
FPS = 30
_max_led_FPS = int(((NUM_LEDS * 30e-6) + 50e-6)**-1.0)
assert FPS <= _max_led_FPS, 'FPS must be <= {}'.format(_max_led_FPS)

"""Frequencies below this value will be removed during audio processing"""
MIN_FREQUENCY = 200

"""Frequencies above this value will be removed during audio processing"""
MAX_FREQUENCY = 12000

"""Number of frequency bins to use when transforming audio to frequency domain
Fast Fourier transforms are used to transform time-domain audio data to the
frequency domain. The frequencies present in the audio signal are assigned
to their respective frequency bins. This value indicates the number of
frequency bins to use.
A small number of bins reduces the frequency resolution of the visualization
but improves amplitude resolution. The opposite is true when using a large
number of bins. More bins is not always better!
There is no point using more bins than there are pixels on the LED strip.
"""

N_FFT_BINS = 24

"""Number of past audio frames to include in the rolling window"""
N_ROLLING_HISTORY = 2

"""No music visualization displayed if recorded audio volume below threshold"""
MIN_VOLUME_THRESHOLD = 1e-7
