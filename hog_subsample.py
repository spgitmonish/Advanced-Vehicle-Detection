import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2

from hog_features import *
from spatial_color_features import *

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, parameter_tuning_dict, debug=False):
    # Make a copy of the image
    draw_img = np.copy(img)

    # Get the area in focus and convert that area from RGB to YCrCb, since
    # the model was optimized for that image format
    img_to_search = img[ystart:ystop,:,:]
    trans_img_to_search = cv2.cvtColor(img_to_search, cv2.COLOR_RGB2YCrCb)

    # The idea with this section is to make sure that the scale of the window
    # is accounted for in the stepping algorithm below. The algorithm doesn't
    # do anything special when different window scales are used. Hurray!
    if scale != 1:
        imshape = trans_img_to_search.shape
        if debug == True:
            print("Original Shape: ", imshape)

        trans_img_to_search = cv2.resize(trans_img_to_search, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        if debug == True:
            print("Transformed Shape: ", trans_img_to_search.shape)

    # Get the individual channels of the image
    ch1 = trans_img_to_search[:,:,0]
    ch2 = trans_img_to_search[:,:,1]
    ch3 = trans_img_to_search[:,:,2]

    # NOTE: '//' operator will do an integer division instead of
    #       floating point division which can be achieved by using
    #       '/' operator instead

    # Define blocks and steps as above
    pix_per_cell = parameter_tuning_dict['pix_per_cell']
    orient = parameter_tuning_dict['orient']
    cell_per_block = parameter_tuning_dict['cell_per_block']
    spatial_size = parameter_tuning_dict['spatial_size']
    hist_bins = parameter_tuning_dict['hist_bins']

    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1

    if debug == True:
        print("Scale: ", scale)
        print("X Blocks: ", nxblocks)
        print("Y Blocks:", nyblocks)

    nfeat_per_block = orient*cell_per_block**2

    if debug == True:
        print("Features per Block: ", nfeat_per_block)

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    if debug == True:
        print("X Steps: ", nxsteps)
        print("Y Steps: ", nysteps)
        print()

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # List of detected windows
    window_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(trans_img_to_search[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                # Calculate window position and append to the list
                # NOTE 1 : The y offset also needs to be factored in.
                # NOTE 2 : The scaling needs to be factored in here
                startx = np.int(xleft*scale)
                endx = np.int(xleft*scale + window*scale)
                starty = np.int(ytop*scale + ystart)
                endy = np.int(ytop*scale + window*scale + ystart)

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))

    return window_list
