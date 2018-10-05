import socket
import mido

from mido import Message


def receive():
    HOST = socket.gethostname() 
    PORT = 5555 
    print(HOST)

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
                    d = int.from_bytes(data, byteorder='little')
                    execute_midi(d)

def execute_midi(note):
    """ execute midi """
    portname = mido.get_output_names()[2]

    try:
        with mido.open_output(portname, autoreset=True) as port:
            print("Using {}".format(port))
            while True:
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
