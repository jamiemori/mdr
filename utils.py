from random import randint


class Rand:
    def __init__(self):
        self.last = None

    def __call__(self):
        r = randint(0, 7)
        while r == self.last:
            r = random.randint(0, 7)
        self.last = r
        return r
