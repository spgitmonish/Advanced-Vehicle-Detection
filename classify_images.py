import numpy as np
import cv2
import glob2
import time
import os
import pickle

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from extract_features import *

def classify_images(parameter_tuning_dict):
    try:
        saved_model_pickle = pickle.load(open("saved_model_pickle.p", "rb"))
        svc = saved_model_pickle["svc"]
        X_scaler = saved_model_pickle["X_scaler"]
    except (OSError, IOError) as e:
        cars = []
        notcars = []

        # Read in cars and notcars
        '''images = glob2.glob('labelled_data/*.jpeg')
        for image in images:
            if 'image' in image or 'extra' in image:
                notcars.append(image)
            else:
                cars.append(image)'''

        # Using glob2's glob API read all the images of
        # vehicles and non-vehicles
        car_images = glob2.glob('Datasets/vehicles/**/*.png')
        for car_image in car_images:
            cars.append(car_image)

        not_car_images = glob2.glob('Datasets/non-vehicles/**/*.png')
        for not_car_image in not_car_images:
            notcars.append(not_car_image)

        # Limit the number of images to 500
        '''sample_size = 500
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]'''

        # Min and max in y to search in slide_window()
        y_start_stop = [300, 700]

        # Extract features of car and not-car images
        car_features = extract_features(cars, parameter_tuning_dict)
        notcar_features = extract_features(notcars, parameter_tuning_dict)

        # Vertically stack all the features to for X(independent dataset)
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)

        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        #rand_state = np.random.randint(0, 100)
        rand_state = 42
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:',
              parameter_tuning_dict['color_space'], 'color space',
              parameter_tuning_dict['orient'], 'orientations',
              parameter_tuning_dict['pix_per_cell'], 'pixels per cell and',
              parameter_tuning_dict['cell_per_block'], 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        # Use a linear SVC
        svc = LinearSVC()

        # Check the training time for the SVC
        t1 = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t1, 2), 'Seconds to train SVC...')

        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        saved_model_pickle = {}
        saved_model_pickle["svc"] = svc
        saved_model_pickle["X_scaler"] = X_scaler
        pickle.dump(saved_model_pickle, open("saved_model_pickle.p", "wb"))

    # Return the ML model and scaler
    return svc, X_scaler
