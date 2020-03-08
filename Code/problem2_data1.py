# header files
import numpy as np
import glob
import sys
from utils import *

# set data path
camera_matrix = np.matrix([[9.037596e+02, 0.000000e+00, 6.957519e+02], [0.000000e+00, 9.019653e+02, 2.242509e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]])
dist_matrix = np.matrix([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])
args = sys.argv
path_images = ""
if(len(args) > 1):
    path_images = args[1]
    files = glob.glob(str(path_images) + "/*")


# read files
for file in files:
    # read frame
    frame = cv2.imread(file)
    
    # pre-processing
    updated_frame = preprocess_image(frame, camera_matrix, dist_matrix)

    # birds-eye view
    (warped_frame, inv_homography_matrix) = get_birds_eye_view(updated_frame)
    
    # get left and right lane
    (intermediate_warped_frame, left_lane_fit, right_lane_fit, _, _) = get_left_and_right_lane(warped_frame, [], [])

    # get radius of curvature, position from center and direction
    (radius_of_curvature, position_from_center, text) = get_information_from_image(left_lane_fit, right_lane_fit)
    
    # get colored warped image
    color_warped_frame = get_color_warp_image(warped_frame, left_lane_fit, right_lane_fit)
    
    # get result image and write information
    result_image = get_result_image(frame, color_warped_frame, inv_homography_matrix)    
    cv2.putText(result_image, 'Vehicle is %.2fm %s of center' % (np.absolute(position_from_center), text), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
    cv2.putText(result_image, 'Radius of Curvature = %.2fm' % radius_of_curvature, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # write image
    #cv2.imwrite("/home/arpitdec5/Desktop/enpm673/projects/Project2/data/data_1/output1/" + str(file.split('/')[-1]), result_image)
