import socket
import mido

from mido import Message


def receive():
    UDP_IP = "108.46.37.32"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(MESSAGE, (UDP_IP, UDP_PORT))
    while True:
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        print("received message:" data)


def execute_midi():
    """ execute midi """
    portname = mido.get_output_names()[2]
    notes = list(range(5, 30))

    try:
        with mido.open_output(portname, autoreset=True) as port:
            print("Using {}".format(port))
            while True:
                for note in notes:
                    on = Message("note_on", note=note)
                    print("Sending {}".format(on))
                    port.send(on)

                    off = Message("note_off", note=note)
                    print("Sending {}".format(off))
                    time.sleep(2.35)
                    port.send(off)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    receive()
