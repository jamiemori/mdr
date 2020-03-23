# import platform
import numpy as np
import config
import opc
import fastopc
import pdb

# Pixel values that were most recently displayed on the LED strip
_prev_pixels = np.tile(253, (config.NUM_STRIPS, config.NUM_LEDS))
pixels = np.tile(1, (config.NUM_STRIPS, config.NUM_LEDS))

# client = opc.Client("localhost:7890")
# client = fastopc.Client("localhost:7890")

def _update_pi():
    """
    Writes new LED values to the Raspberry Pi's LED strip
    Raspberry Pi uses the rpi_ws281x to control the LED strip directly.
    This function updates the LED strip with new values.
    """
    global pixels, _prev_pixels

    # Truncate values and cast to integer
    pixels = np.clip(pixels, 0, 255).astype(int)
    x = np.copy(pixels)

    # # Encode 24-bit LED values in 32 bit integers
    # r = np.left_shift(p[0][:].astype(int), 8)
    # g = np.left_shift(p[1][:].astype(int), 16)
    # b = p[2][:].astype(int)
    # rgb = np.bitwise_or(np.bitwise_or(r, g), b)

    # # Update the pixels
    # for i in range(config.NUM_LEDS):
        # # Ignore pixels if they haven't changed (saves bandwidth)
        # if np.array_equal(p[:, i], _prev_pixels[:, i]):
            # continue
        # # strip._led_data[i] = rgb[i]

    print(x)
    print(x.shape)
    _prev_pixels = np.copy(x)

def update():
    _update_pi()

# Execute this file to run a LED strand test
# If everything is working, you should see a red, green, and blue pixel scroll
# across the LED strip continously
if __name__ == '__main__':
    import time

    # Turn all pixels off
    pixels *= 0
    pixels[0, 0] = 255  # Set 1st pixel red
    pixels[1, 1] = 255  # Set 2nd pixel green
    pixels[2, 2] = 255  # Set 3rd pixel blue
    print('Starting LED strand test')

    while True:
        pixels = np.roll(pixels, 1, axis=1)
        update()
        time.sleep(.1)
