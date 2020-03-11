"""
 *  MIT License
 *
 *  Copyright (c) 2019 Arpit Aggarwal Shantam Bajpai
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without
 *  limitation the rights to use, copy, modify, merge, publish, distribute,
 *  sublicense, and/or sell copies of the Software, and to permit persons to
 *  whom the Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
"""

# header files
from utils import *
import sys

# define constants
args = sys.argv
path_video = ""
if(len(args) > 1):
    path_video = args[1]
cap = cv2.VideoCapture(str(path_video))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("night_drive.avi", fourcc, 20.0, (1920, 1080))

# gamma map
gamma_map = {}
for value in range(0, 256):
    gamma_map[value] = int(((value/255)**(1/2))*255)

# read video
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret):
        # get current video frame
        height, width = frame.shape[0], frame.shape[1]
        frame = cv2.addWeighted(frame, 2, frame, 0, 2)

        # preprocess using gamma correction
        for row in range(0, height):
            for col in range(0, width):
                frame[row, col, 0] = gamma_map[frame[row, col, 0]]
                frame[row, col, 1] = gamma_map[frame[row, col, 1]]
                frame[row, col, 2] = gamma_map[frame[row, col, 2]]

        out.write(frame)
        count = count + 1
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
