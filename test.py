# All other necessary modules
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob2

# Scipy image reading function
from scipy import misc
from scipy.ndimage.measurements import label
from scipy.ndimage.measurements import center_of_mass

# Import local functions
from spatial_color_features import *
from hog_features import *
from sliding_window import *
from extract_features import *
from apply_heat import *
from hog_subsample import *
from classify_images import *

# Import movie editor files
from moviepy.editor import VideoFileClip

# Dictionary for all the parameters which can be tuned/changed
global parameter_tuning_dict
parameter_tuning_dict = {
    'color_space' : 'YCrCb', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient' : 9, # HOG orientations
    'pix_per_cell' : 8, # HOG pixels per cell
    'cell_per_block' : 2, # HOG cells per block
    'hog_channel' : "ALL", # Can be 0, 1, 2, or "ALL"
    'spatial_size' : (32, 32), # Spatial binning dimensions
    'hist_bins' : 32, # Number of histogram bins
    'spatial_feat' : True, # Spatial features on or off
    'hist_feat' : True, # Histogram features on or off
    'hog_feat' : True, # HOG features on or off
}

def pipelineVideo(image):
    # The following are global variables which are defined in the caller
    global list_of_boxes_list
    global frame_count
    global svc
    global X_scaler

    # Copy of the image to draw on
    draw_image = np.copy(image)

    # To display boxes for debug
    display_boxes = False

    # The area in focus
    ystart = 400
    ystop = 656

    scales = [0.6, 0.75, 1.0]
    #scales = [0.75, 1.0]
    #scales = [0.6, 0.7, 0.8, 0.9, 1.0]
    #scales = [1.0]

    # This code can be modified to process every other frame instead of just
    # one frame at a time(which can be very time consuming)
    if frame_count % 1 == 0:
        for scale in scales:
            detected_windows = find_cars(image, ystart, ystop, scale,
                                         svc, X_scaler, parameter_tuning_dict)
            # This means that there is no car in the picture, remove all the entries
            # in the FIFO because we don't want any false detections based on the
            # old FIFO entries
            if detected_windows:
                # Append the latest window detected to the
                list_of_boxes_list.append(detected_windows)
                if len(list_of_boxes_list) > 4:
                    # Pop the oldest entry to keep the list fresh
                    list_of_boxes_list.pop(0)
            else:
                if list_of_boxes_list:
                    # Pop the oldest entry
                    list_of_boxes_list.pop(0)

            if display_boxes == True:
                window_img = draw_boxes(draw_image, detected_windows,
                                        color=(0, 0, 255), thick=6)

                plt.imshow(window_img)
                plt.show()

    # Threshold has been reached, use the existing list of boxes, create a
    # heat map and then add windows around the vehicles
    if len(list_of_boxes_list) >= 4:
        # Image for adding heat(use the last image)
        heat = np.zeros_like(image[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, list_of_boxes_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 5)

        # Visualize the heatmap when displaying
        # NOTE: Limit the values from 0<->255
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        # Find the center of mass for the labels generated
        draw_image = draw_labeled_bboxes(np.copy(image), labels)

        if display_boxes == True:
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            plt.subplot(122)
            plt.imshow(draw_image)
            plt.title('Car Positions')
            plt.show()

    # Increment frame count
    frame_count = frame_count + 1

    # Return the original image if no detections were found or the modified
    # image with boxes around the vehicles(based on the heat map)
    return draw_image

# Call function which classifies images and returns the model and the
# scaler object fit for car and not-car images
svc, X_scaler = classify_images(parameter_tuning_dict)

# List of boxes
list_of_boxes_list = []

# Global count to skip frames to reduce processing time
frame_count = 0

project_output = 'project_video_output.mp4'
project_clip = VideoFileClip("project_video.mp4")
project_clip = project_clip.fl_image(pipelineVideo)
project_clip.write_videofile(project_output, audio=False)
