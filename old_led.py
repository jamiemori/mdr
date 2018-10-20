"""
Old code for the interactive touch element of hikari no hako
"""

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

