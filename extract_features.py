import numpy as np
import cv2
import os

from scipy import misc
from hog_features import *
from spatial_color_features import *

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# NOTE: just for a single image rather than list of images
def single_img_features(img, parameter_tuning_dict):
    #1) Define an empty list to receive features
    img_features = []

    #2) Apply color conversion if other than 'RGB'
    color_space = parameter_tuning_dict['color_space']

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif color_space == 'Lab':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

        # The training was done on a scale of 0-1 but cv2.cvtColor
        # resizes after conversion to 0-255, so convert back
        #feature_image = feature_image.astype(np.float32)/255
    else:
        # Use the original image format
        feature_image = np.copy(img)

    #3) Compute spatial features if flag is set
    if parameter_tuning_dict['spatial_feat'] == True:
        spatial_features = bin_spatial(feature_image, size=parameter_tuning_dict['spatial_size'])

        #4) Append features to list
        img_features.append(spatial_features)

    #5) Compute histogram features if flag is set
    if parameter_tuning_dict['hist_feat'] == True:
        hist_features = color_hist(feature_image, nbins=parameter_tuning_dict['hist_bins'])

        #6) Append features to list
        img_features.append(hist_features)

    #7) Compute HOG features if flag is set
    if parameter_tuning_dict['hog_feat'] == True:
        if parameter_tuning_dict['hog_channel'] == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                # NOTE: Extend the list with the element provided
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    parameter_tuning_dict['orient'],
                                    parameter_tuning_dict['pix_per_cell'],
                                    parameter_tuning_dict['cell_per_block'],
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel],
                                            parameter_tuning_dict['orient'],
                                            parameter_tuning_dict['pix_per_cell'],
                                            parameter_tuning_dict['cell_per_block'],
                                            vis=False, feature_vec=True)

        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, parameter_tuning_dict):
    # Create a list to append feature vectors to(all images)
    features = []

    # Iterate through the list of images
    for file in imgs:
        # Features list for the file
        file_features = []

        # Read in each one by one
        #image = mpimg.imread(file)
        image = misc.imread(file)

        # Local copy of the color space to be used
        color_space = parameter_tuning_dict['color_space']

        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            elif color_space == 'Lab':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        else:
            feature_image = np.copy(image)

        if parameter_tuning_dict['spatial_feat'] == True:
            spatial_features = bin_spatial(feature_image, size=parameter_tuning_dict['spatial_size'])
            file_features.append(spatial_features)
        if parameter_tuning_dict['hist_feat'] == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=parameter_tuning_dict['hist_bins'])
            file_features.append(hist_features)
        if parameter_tuning_dict['hog_feat'] == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if parameter_tuning_dict['hog_channel'] == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        parameter_tuning_dict['orient'],
                                        parameter_tuning_dict['pix_per_cell'],
                                        parameter_tuning_dict['cell_per_block'],
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel],
                                                parameter_tuning_dict['orient'],
                                                parameter_tuning_dict['pix_per_cell'],
                                                parameter_tuning_dict['cell_per_block'],
                                                vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)

        # Append all the features
        features.append(np.concatenate(file_features))

    # Return list of feature vectors
    return features
