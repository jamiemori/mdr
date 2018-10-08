import socket
import mido
import time
import pdb

import threading
import queue

from mido import Message

midi_queue = queue.Queue()


def midi_worker(midi_queue):
    while True:
        note = midi_queue.get()
        print('current queue size ---- ', midi_queue.qsize())
        execute_midi(note)
        midi_queue.task_done()


def execute_midi(note):
    """ execute midi """
    portname = mido.get_output_names()[2]
    try:
        with mido.open_output(portname) as port:
            print("Using {}".format(port))

            on = Message("note_on", note=note)
            print("Sending {}".format(on))
            port.send(on)

            print(threading.get_ident())
            for i in range(3):
                print(i, threading.get_ident())
                time.sleep(1)
                i -= 1
            off = Message("note_off", note=note)
            print("Sending {}".format(off))
            port.send(off)
    except KeyboardInterrupt:
        pass


def receive():
    """ receive socket message from sender """
    HOST = socket.gethostname()
    PORT = 5555

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()

    worker_threads = []
    for i in range(20):
        t = threading.Thread(target=midi_worker, daemon=True, args=(midi_queue,))
        worker_threads.append(t)
        t.start()

    while True:
        print("waiting for connection")
        conn, addr = s.accept()
        try:
            print("connection from", addr)
            while True:
                data = conn.recv(16)
                print("received {!r}".format(data))
                if not data:
                    break
                else:
                    d = int.from_bytes(data, byteorder="little")
                    print(d)
                    midi_queue.put(d)
        except KeyboardInterrupt:
            raise
        finally:
            conn.close()


if __name__ == "__main__":
    print("socket opened")
    receive()
