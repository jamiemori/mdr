import socket
import mido

from mido import Message


def receive():
    HOST = '192.168.1.223'
    PORT = 5555 

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                else:
                    print(data)

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


if __name__ == "__main__":
    print('socket opened')
    receive()
