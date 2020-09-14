import cv2
import numpy as np
import argparse

'''
Command line guide:
python circle_detecion.py --image <Image Path>
'''

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                required=True)
args = vars(ap.parse_args())

# Trackbar
def nothing(x):
    pass

cv2.namedWindow('image')

'''
Create Trackbar for parameter changing
The two numbers represent lower and upper limit of trackbar
'''
# This is how strong the detected edges have to be considered an edge of a circle
cv2.createTrackbar('Circle Edge Strength', 'image', 50, 150, nothing)
# This is the minimum number of edge points to declare object as a circle
cv2.createTrackbar('Edge Point To Declare Circle', 'image', 30, 100, nothing)
# Minimum radius of object to be considered circle
cv2.createTrackbar('Minimum Radius', 'image', 1, 50, nothing)
# Maximum radius of object to be considered circle
cv2.createTrackbar('Maximum Radius', 'image', 40, 150, nothing)

while True:
    # Read image
    img = cv2.imread(args["image"], cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur with 21x21 kernel
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), cv2.BORDER_DEFAULT)


    # Get variables from trackbar
    param1 = cv2.getTrackbarPos('Circle Edge Strength', 'image')
    param2 = cv2.getTrackbarPos('Edge Point To Declare Circle', 'image')
    minRadius = cv2.getTrackbarPos('Minimum Radius', 'image')
    maxRadius = cv2.getTrackbarPos('Maximum Radius', 'image')
    
    # Apply Hough Transform
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1 = param1,
                                        param2 = param2, minRadius= minRadius, maxRadius= maxRadius)

    # Draw detected circles
    if detected_circles is not None:

        # Convert circle parameters to integer
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw circle circumference
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

print(img.shape[0])
print(img.shape[1])
cv2.destroyAllWindows()