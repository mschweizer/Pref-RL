import random
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc


class SegmentRenderer:

    def __init__(self, output_path):
        self.output_path = output_path

    def render_segment(self, segment):
        outfile = self.output_path + str(random.randint(0, 100000000)) + '.avi'
        fps = 12
        fourcc = VideoWriter_fourcc(*'DIVX')
        singleframe = np.array(segment.frames[0])
        fshape = (singleframe.shape[1], singleframe.shape[0])

        vid_writer = VideoWriter(outfile, fourcc, fps, fshape)
        
        for frame in segment.frames:
            vid_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        vid_writer.release()