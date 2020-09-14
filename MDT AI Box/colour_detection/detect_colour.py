import numpy as np
import argparse
import cv2
import sys
from non_max_suppression import remove_overlapping

'''
Command line guide:
python detect_colour.py --image <Image Path> --colour <Colour to detect>
Colours accepted so far: Red, Blue, Green, Yellow
<Note: Unable to calibrate too many colours as colour boundary will overlap>
'''

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
ap.add_argument("-c", "--colour", required=True, help="colour to detect")
args = vars(ap.parse_args())

# Initialise colour boundaries for thresholding
boundaries = [
    ([0, 0, 110], [70, 65, 200]), # Red 1
    ([0, 0, 200], [115, 120, 255]), # Red 2
    ([86, 10, 0], [255, 220, 70]), # Blue
    ([0, 60, 0], [160, 255, 60]), # Green 1
    ([65, 160, 100], [170, 255, 226]), # Green 2
    ([0, 145, 195], [125, 255, 255]) # Yellow
]

# Perform checks on command line input to make sure colour specified is acceptable
colour = args["colour"].lower()
if (colour != "red") and (colour != "green") and (colour != "blue") and (colour != "yellow"):
    print("Please specify a colour that is either red, green, blue, or yellow")
    sys.exit()

# Load image
image = cv2.imread(args["image"])
width = image.shape[1]
height = image.shape[0]
if (width > 1000) and (height > 1000):
    image = cv2.resize(image, (int(width / 4), int(height / 4)))

# Determine upper and lower RGB boundaries as well as threshold value
if colour == "red":
    lower1 = np.array(boundaries[0][0], dtype="uint8")
    upper1 = np.array(boundaries[0][1], dtype="uint8")
    lower2 = np.array(boundaries[1][0], dtype="uint8")
    upper2 = np.array(boundaries[1][1], dtype="uint8")
    threshold_val = 25
    color = (0, 255, 0)
elif colour == "blue":
    lower = np.array(boundaries[2][0], dtype="uint8")
    upper = np.array(boundaries[2][1], dtype="uint8")
    threshold_val = 35
    color = (0, 0, 255)
elif colour == "yellow":
    lower = np.array(boundaries[5][0], dtype="uint8")
    upper = np.array(boundaries[5][1], dtype="uint8")
    threshold_val = 25
    color = (0, 0, 255)
else:
    lower1 = np.array(boundaries[3][0], dtype="uint8")
    lower2 = np.array(boundaries[4][0], dtype="uint8")
    upper1 = np.array(boundaries[3][1], dtype="uint8")
    upper2 = np.array(boundaries[4][1], dtype="uint8")

    # Special case when yellow boundary needed too (Too similar to green)
    yellow_lower = np.array(boundaries[5][0], dtype="uint8")
    yellow_upper = np.array(boundaries[5][1], dtype="uint8")

    color = (0, 0, 255)
    threshold_val = 10


# Find colors within specified boundaries and apply mask (Green special case)
if colour == "green":
    # Create yellow mask
    yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)
    output_yellow = cv2.bitwise_and(image, image, mask = yellow_mask)
    # Remove yellow from image first so no interference with green
    image_copy = image - output_yellow
    cv2.imshow('img', image)

    # Create green masks
    mask1 = cv2.inRange(image_copy, lower1, upper1)
    mask2 = cv2.inRange(image_copy, lower2, upper2)
    output1 = cv2.bitwise_and(image_copy, image_copy, mask = mask1)
    output2 = cv2.bitwise_and(image_copy, image_copy, mask = mask2)
    # Combine two masked images (dark and light green)
    output = output1 + output2
    cv2.imshow('output', output)
elif colour == "red":
    mask1 = cv2.inRange(image, lower1, upper1)
    mask2 = cv2.inRange(image, lower2, upper2)
    output1 = cv2.bitwise_and(image, image, mask=mask1)
    output2 = cv2.bitwise_and(image, image, mask=mask2)
    output = output1 + output2
    cv2.imshow('output', output)
else:
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    cv2.imshow('output', output)

# Grayscale and threshold image
output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(output_gray, threshold_val, 255, 0)
# Denoise image
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (4, 4))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (4, 4))

# Image Processing Checkpoint
cv2.imshow('thresh', thresh)

# Detect image contours for drawing bounding box later
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialise box list for NMS (Experimental, doesn't work so far)
boxes = []

for i in range(len(contours)):
    cnt = contours[i]
    (x, y, w, h) = cv2.boundingRect(cnt)
    box = (x, y, x+w, y+h)

    # Filter out small boxes (sometimes due to noise or inaccurate colour thresholding)
    if (w > 30) and (h > 30):
        boxes.append(box)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

# convert bounding box list to numpy array
boxes = np.array(boxes)

# Get boxes to draw (remove_overlapping is non_max_suppression function)
# pick = remove_overlapping(boxes, 0.5)

# Draw singular boxes for NMS version
# for (x1, y1, w, h) in pick:
#     cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

# Show images
cv2.imshow('image', image)
k = cv2.waitKey(0) & 0xFF
if k == 27:
    sys.exit()