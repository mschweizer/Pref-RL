from abc import ABC, abstractmethod
import cv2
import random
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc, randn
from pathlib import Path


class SegmentRenderer:

    def __init__(self, out):
        self.out = out

    def render_segment(self, query):
        outfile = self.out + str(random.randint(0, 100000000)) + '.avi'
        fps = 8
        fourcc = VideoWriter_fourcc(*'DIVX')
        singleframe = np.array(query.frames[0])
        fshape = singleframe.shape
        vid_writer = VideoWriter(outfile, fourcc, fps, fshape, False)

        for frame in query.frames:
            vid_writer.write(frame)

        vid_writer.release()    
        
        self.nr = self.nr + 1
