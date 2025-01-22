#!/usr/win/pyhton 3.6 by DZN


class Antenna:
    def __init__(self, rgain=10.0**(1 / 10), tgain=1.0):
        self.rgain = rgain
        self.tgain = tgain
