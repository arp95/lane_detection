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
camera_matrix = np.matrix([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_matrix = np.matrix([-2.42565104e-01, -4.77893070e-02 , -1.31388084e-03 , -8.79107779e-05, 2.20573263e-02])
out = cv2.VideoWriter("output_challenge_video.avi", fourcc, 20.0, (1280, 720))

# read video
count = 0
prev_right_lane_fit = []
prev_left_lane_fit = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret):
        # get current video frame
        height, width = frame.shape[0], frame.shape[1]
        # pre-processing
        updated_frame = preprocess_image(frame, camera_matrix, dist_matrix)
        
        # birds-eye view
        (warped_frame, inv_homography_matrix) = get_birds_eye_view_video(updated_frame)
    
        # get left and right lane
        (intermediate_warped_frame, left_lane_fit, right_lane_fit, length1, length2) = get_left_and_right_lane(warped_frame, prev_left_lane_fit, prev_right_lane_fit)
        if(length1 > 4):
            prev_left_lane_fit = left_lane_fit
        if(length2 > 4):
            prev_right_lane_fit = right_lane_fit
    
        # get radius of curvature, position from center and direction
        (radius_of_curvature, position_from_center, text) = get_information_from_image(left_lane_fit, right_lane_fit)
    
        # get colored warped image
        color_warped_frame = get_color_warp_image(warped_frame, left_lane_fit, right_lane_fit)
    
        # get result image and write information
        result_image = get_result_image(frame, color_warped_frame, inv_homography_matrix)    
        cv2.putText(result_image, 'Vehicle is %.2fm %s of center' % (np.absolute(position_from_center), text), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
        cv2.putText(result_image, 'Radius of Curvature = %.2fm' % radius_of_curvature, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(result_image)
        count = count + 1
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
