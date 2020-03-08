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
