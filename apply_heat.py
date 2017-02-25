import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2

def add_heat(heatmap, list_of_boxes_list):
    # Parse the list of boxes and add heat
    for box_list in list_of_boxes_list:
        # Here might lie the entire logic
        # Iterate through list of bboxes
        for box in box_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 2.5

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, mass_center):
    new_bbox = []
    new_mass_center = []
    car_labels = list(range(1, labels[1] + 1))

    # NOTE: The assumption made here is that there are utmost 2 boxes close to
    #       each other and should be indicating a single car
    if len(mass_center) > 1:
         for index in range(1, len(mass_center)):
             # Check if the boxes are very close to each other, this most likely
             # means that it is the same car
             if ((abs(mass_center[index - 1][0] - mass_center[index][0]) <= 10) and
                 (abs(mass_center[index - 1][1] - mass_center[index][1]) <= 100)):
                 # Find pixels with each car_number label value
                 nonzero_1 = (labels[0] == car_labels[index-1]).nonzero()

                 # Identify x and y values of those pixels
                 nonzero_y_1 = np.array(nonzero_1[0])
                 nonzero_x_1 = np.array(nonzero_1[1])

                 # Define a bounding box based on min/max x and y
                 bbox_1 = np.array([[np.min(nonzero_x_1), np.min(nonzero_y_1)],
                                    [np.max(nonzero_x_1), np.max(nonzero_y_1)]])

                 # Find pixels with each car_number label value
                 nonzero_2 = (labels[0] == car_labels[index]).nonzero()

                 # Identify x and y values of those pixels
                 nonzero_y_2 = np.array(nonzero_2[0])
                 nonzero_x_2 = np.array(nonzero_2[1])

                 # Define a bounding box based on min/max x and y
                 bbox_2 = np.array([[np.min(nonzero_x_2), np.min(nonzero_y_2)],
                                    [np.max(nonzero_x_2), np.max(nonzero_y_2)]])
                 
                 # Pixel positions are integers
                 x1 = (bbox_1[0][0] + bbox_2[0][0]) // 2
                 y1 = (bbox_1[0][1] + bbox_2[0][1]) // 2
                 x2 = (bbox_1[1][0] + bbox_2[1][0]) // 2
                 y2 = (bbox_1[1][1] + bbox_2[1][1]) // 2

                 # New box based on the average
                 bbox = [(x1, y1), (x2, y2)]

                 # Append to the list
                 new_bbox.append(bbox)

                 center_of_mass = np.mean((mass_center[index - 1], mass_center[index]), axis=0)
                 new_mass_center.append(center_of_mass)

                 # Increment index to account for a comparison already done
                 index = index + 1
             else:
                 # Find pixels with each car_number label value
                 nonzero = (labels[0] == car_labels[index-1]).nonzero()

                 # Identify x and y values of those pixels
                 nonzero_y = np.array(nonzero[0])
                 nonzero_x = np.array(nonzero[1])

                 # Define a bounding box based on min/max x and y
                 bbox = ((np.min(nonzero_x), np.min(nonzero_y)),
                         (np.max(nonzero_x), np.max(nonzero_y)))

                 new_bbox.append(bbox)

         # Iterate through all the boxes to be drawn
         for bbox in new_bbox:
             # Draw the box on the image
             cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Single hot spot
    else:
        # Iterate through all detected labelled cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    # Return the image
    return img
