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

def draw_labeled_bboxes(img, labels):
    boxes_list = []
    debug = False
    not_enhanced = False

    # Iterate through all detected labelled cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Append the boxes
        boxes_list.append(bbox)

    if not_enhanced == True:
        for box in boxes_list:
            # Draw the box on the image
            cv2.rectangle(img, box[0], box[1], (0,0,255), 6)
    else:
        if debug == True:
            print(boxes_list)
            print()

        # List which gets marked with 1 if
        # 1. No hit against any box
        # 2. Hit against a box and should be skipped
        boxes_checked_list = [0]*len(boxes_list)

        # Store the new boxes list after comparison
        updated_boxes_list = []

        # Go through all the boxes in the list and compare
        for box_index in range(len(boxes_list)):
            # Box 1
            box_1 = np.array(boxes_list[box_index])
            if debug == True:
                print("Box1")
                print(box_1)

            # If the box area is very small then ignore the box(most likely false)
            if((abs(box_1[0][0] - box_1[1][0]) <= 40) or
               (abs(box_1[0][1] - box_1[1][1]) <= 40)):
               # Marked as compared already
               boxes_checked_list[box_index] = 1

            # If the box has been compared already then continue to next iteration
            if boxes_checked_list[box_index] == 1:
                continue

            # Get next box to compare
            comparison_index = box_index + 1

            # Compare against all next boxes till the end
            while comparison_index < len(boxes_list):
                # Get the box to compare
                box_2 = np.array(boxes_list[comparison_index])
                if debug == True:
                    print()
                    print("Box2")
                    print(box_2)

                # Make sure this box hasn't been compared against before
                if boxes_checked_list[comparison_index] == 0:
                    # 1. Check if x1 of Box2 is very close to x2 of Box1
                    # 2. Check if y2 of both boxes are in the same area
                    if((abs(box_2[0][0] - box_1[1][0]) <= 40) and
                       (abs(box_1[1][1] - box_2[1][1]) <= 75)):

                        # Mark that Box2 has been compared already
                        boxes_checked_list[comparison_index] = 1

                        # The update x2 of Box1 to be equal to x2 of Box2
                        box_1[1][0] = box_2[1][0]

                        # Also update y2 of Box1 if y2 of Box2 is higher
                        if(box_2[1][1] > box_1[1][1]):
                            box_1[1][1] = box_2[1][1]

                        if debug == True:
                            print("X Hit")
                            print("Updated Box1")
                            print(box_1)

                    # 1. Check if y1 of Box2 is very close to y2 of Box1
                    # 2. Check if x2 of both boxes are in the same area
                    if((abs(box_2[0][1] - box_1[1][1]) <= 40) and
                       (abs(box_1[1][0] - box_2[1][0]) <= 75)):

                        # Mark that Box2 has been compared already
                        boxes_checked_list[comparison_index] = 1

                        # The update y2 of Box1 to be equal to y2 of Box2
                        box_1[1][1] = box_2[1][1]

                        # Also update x2 of Box1 if x2 of Box2 is higher
                        if(box_2[1][0] > box_1[1][0]):
                            box_1[1][0] = box_2[1][0]

                        if debug == True:
                            print("Y Hit")
                            print("Updated Box1")
                            print(box_1)

                # Increment count
                comparison_index = comparison_index + 1

            # Add box_1 to the list of new boxes to draw
            updated_boxes_list.append(box_1.tolist())

            # Mark that Box2 has finished comparison
            boxes_checked_list[box_index] = 1

        if debug == True:
            print("Updated Boxes")
            print(updated_boxes_list)

        for index in range(len(updated_boxes_list)):
            box = updated_boxes_list[index]

            # Set up the vertices in the correct format
            draw_box = ((box[0][0], box[0][1]),
                        (box[1][0], box[1][1]))

            # Draw the box on the image
            cv2.rectangle(img, draw_box[0], draw_box[1], (0,0,255), 6)

    # Return the image
    return img
