import socket
import mido
import time
import pdb

from mido import Message


def receive():
    """ receive socket message from sender """
    HOST = socket.gethostname() 
    PORT = 5555 

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()

    while True:
        print('waiting for connection')
        conn, addr = s.accept()
        try:
            print('connection from', addr)
            while True:
                data = conn.recv(1024)
                print('received {!r}'.format(data))
                if not data:
                    break
                else:
                    d = int.from_bytes(data, byteorder='little')
                    print(d)
                    execute_midi(d)
        finally:
            conn.close()


def execute_midi(note):
    """ execute midi """
    portname = mido.get_output_names()[2]

    try:
        with mido.open_output(portname, autoreset=True) as port:
            print("Using {}".format(port))

            on = Message("note_on", note=note)
            print("Sending {}".format(on))
            port.send(on)

            off = Message("note_off", note=note)
            print("Sending {}".format(off))
            time.sleep(2.35)
            port.send(off)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    print('socket opened')
    receive()
