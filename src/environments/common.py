"""
@author: Francisco Maestre. Modified version of Viet Nguyen
"""

import subprocess as sp

'''
La clase Monitor unicamente s eencarga de arrancar la pantalal en la que vamos a ver el juego funcionando
'''

class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())

