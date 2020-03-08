# header files
import numpy as np
import cv2

# function for preprocessing of image
def preprocess_image(frame, camera_matrix, dist_matrix):
    # undistort the frame
    frame = cv2.undistort(frame, camera_matrix, dist_matrix)
    
    # average blurring
    frame = cv2.blur(frame, (3, 3))
    
    # Convert to HLS color space and apply masks
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS).astype(np.float)
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    lower_yellow = np.array([50, 0, 100], dtype=np.uint8)
    upper_yellow = np.array([50, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)  
    
    # get the binary image
    combined_binary = np.zeros((frame.shape[0], frame.shape[1]))
    count = 0
    for row in range(0, white_mask.shape[0]):
        for col in range(0, white_mask.shape[1]):
            if(white_mask[row, col] >= 200 or yellow_mask[row, col] >= 200):
                combined_binary[row, col] = 255
    return combined_binary

# function for preprocessing of image
def preprocess_image_video(frame, camera_matrix, dist_matrix):
    # undistort the frame
    frame = cv2.undistort(frame, camera_matrix, dist_matrix)
    
    # average blurring
    frame = cv2.blur(frame, (3, 3))
    
    # Convert to HLS color space and apply masks
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS).astype(np.float)
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    lower_yellow = np.array([10, 0, 90], dtype=np.uint8)
    upper_yellow = np.array([30, 220, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)  
    
    # get the binary image
    combined_binary = np.zeros((frame.shape[0], frame.shape[1]))
    count = 0
    for row in range(0, white_mask.shape[0]):
        for col in range(0, white_mask.shape[1]):
            if(white_mask[row, col] >= 210 or yellow_mask[row, col] >= 210):
                combined_binary[row, col] = 255
    return combined_binary

# get birds eye view
def get_birds_eye_view(frame):
    image = np.copy(frame)
    src = np.float32([[586, 293], [720, 293], [867, 507], [255, 507]])
    dst = np.float32([[200, 0], [image.shape[1] - 200, 0], [image.shape[1] - 200, image.shape[0]], [200, image.shape[0]]])
    homography_matrix = cv2.getPerspectiveTransform(src, dst)
    inv_homography_matrix = cv2.getPerspectiveTransform(dst, src)
    image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))
    return (image, inv_homography_matrix)

# get birds eye view
def get_birds_eye_view_video(frame):
    image = np.copy(frame)
    src = np.float32([[572, 510], [789, 510], [1092, 715], [282, 715]])
    dst = np.float32([[200, 0], [image.shape[1] - 200, 0], [image.shape[1] - 200, image.shape[0]], [200, image.shape[0]]])
    homography_matrix = cv2.getPerspectiveTransform(src, dst)
    inv_homography_matrix = cv2.getPerspectiveTransform(dst, src)
    image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))
    return (image, inv_homography_matrix)

# get left and right lane coordinates
def get_left_and_right_lane(frame, prev_left_lane_fit, prev_right_lane_fit):
    # create windows and check left and right lane
    image = np.dstack((frame, frame, frame)) * 255
    histogram_along_col = np.sum(frame[int(frame.shape[0] / 10):,:], axis = 0)
    left_col_max = np.argmax(histogram_along_col[:int(histogram_along_col.shape[0] / 2)])
    right_col_max = np.argmax(histogram_along_col[int(histogram_along_col.shape[0] / 2):]) + int(histogram_along_col.shape[0] / 2)
    left_col_current = left_col_max
    right_col_current = right_col_max
    left_lane_ind = []
    right_lane_ind = []
    
    # iterate through windows and store left lane and right lane coordinates
    for window in range(0, int(frame.shape[0] / 10)):
        window_row_min = frame.shape[0] - ((window + 1) * int(frame.shape[0] / 10))
        window_row_max = frame.shape[0] - (window * int(frame.shape[0] / 10))
        window_col_left_low = left_col_current - 50
        window_col_left_high = left_col_current + 50
        window_col_right_low = right_col_current - 50
        window_col_right_high = right_col_current + 50
        
        cv2.rectangle(image, (window_col_left_low, window_row_min), (window_col_left_high, window_row_max), (0, 255, 0))
        cv2.rectangle(image, (window_col_right_low, window_row_min), (window_col_right_high, window_row_max), (0, 0, 255))
        
        # left lane window
        left_count = 0
        left_sum = 0.0
        for row in range(max(0, window_row_min), min(frame.shape[0], window_row_max)):
            for col in range(max(0, window_col_left_low), min(frame.shape[1], window_col_left_high)):
                if(frame[row, col] > 0):
                    left_count = left_count + 1
                    left_sum = left_sum + col
                    left_lane_ind.append((row, col))
                    image[row, col] = (0, 255, 0)
        if(left_count > 20):
            left_col_current = int(left_sum / left_count)
        
        # right lane window
        right_count = 0
        right_sum = 0.0
        for row in range(max(0, window_row_min), min(frame.shape[0], window_row_max)):
            for col in range(max(0, window_col_right_low), min(frame.shape[1], window_col_right_high)):
                if(frame[row, col] > 0):
                    right_count = right_count + 1
                    right_sum = right_sum + col
                    right_lane_ind.append((row, col))
                    image[row, col] = (0, 0, 255)
        if(right_count > 20):
            right_col_current = int(right_sum / right_count)
            
    # left lane and right lane fit
    left_lane_rows = []
    left_lane_cols = []
    for index in range(0, len(left_lane_ind)):
        left_lane_rows.append(left_lane_ind[index][0])
        left_lane_cols.append(left_lane_ind[index][1])
    
    right_lane_rows = []
    right_lane_cols = []
    for index in range(0, len(right_lane_ind)):
        right_lane_rows.append(right_lane_ind[index][0])
        right_lane_cols.append(right_lane_ind[index][1])
        
    if(len(right_lane_rows) > 4):
        right_lane_fit = np.polyfit(right_lane_rows, right_lane_cols, 2)
    else:
        right_lane_fit = prev_right_lane_fit

    if(len(left_lane_rows) > 4):
        left_lane_fit = np.polyfit(left_lane_rows, left_lane_cols, 2)
    else:
        left_lane_fit = prev_left_lane_fit
    return (image, left_lane_fit, right_lane_fit, len(left_lane_rows), len(right_lane_rows))

# get radius of curvature of lane
def get_information_from_image(left_lane_fit, right_lane_fit):
    ym_per_pix = 2 / 72 
    xm_per_pix = 4 / 660 
    y1 = ((2 * left_lane_fit[0] * 700) + left_lane_fit[1]) * (xm_per_pix / ym_per_pix)
    y2 = (2 * left_lane_fit[0] * xm_per_pix) / (ym_per_pix * ym_per_pix)
    curvature = ((1 + y1 * y1)**(1.5)) / np.absolute(y2)
    
    x_left_pix = left_lane_fit[0] * (700**2) + left_lane_fit[1] * 700 + left_lane_fit[2]
    x_right_pix = right_lane_fit[0] * (700**2) + right_lane_fit[1] * 700 + right_lane_fit[2]
    position_from_center = ((x_left_pix + x_right_pix)/2 - 650) * xm_per_pix
    if position_from_center < 0:
        text = 'left'
    else:
        text = 'right'
    return (curvature, position_from_center, text)

# get the colored warp image of left lane and right lane points
def get_color_warp_image(frame, left_lane_fit, right_lane_fit):
    row_pts = []
    left_pts = []
    right_pts = []
    for row in range(0, frame.shape[0]):
        row_pts.append(row)
        
        # left
        left_pts.append(left_lane_fit[0] * row ** 2 + left_lane_fit[1] * row + left_lane_fit[2])
        
        # right
        right_pts.append(right_lane_fit[0] * row ** 2 + right_lane_fit[1] * row + right_lane_fit[2])
    
    left_pts = np.array(left_pts)
    right_pts = np.array(right_pts)
    warp_zero = np.zeros_like(frame).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_pts, row_pts]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_pts, row_pts])))])
    cv2.fillPoly(color_warp, np.int_([np.hstack((pts_left, pts_right))]), (0, 255, 0))
    return color_warp

# get result image
def get_result_image(frame, color_warped_frame, inv_homography_matrix):
    new_warped_frame = cv2.warpPerspective(color_warped_frame, inv_homography_matrix, (frame.shape[1], frame.shape[0]))
    result_image = cv2.addWeighted(frame, 1, new_warped_frame, 0.5, 0)
    return result_image
