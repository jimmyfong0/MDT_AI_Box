from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import face_recognition

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

def eye_aspect_ratio(eye):
    # compute distance between two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute euclidean distance between horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])

    # compute eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return eye aspect ratio
    return ear

# Declare blink threshold (distance between eye landmark will decrease when blink)
EYE_RATIO_THRESH = 0.25
# Initialise how many consecutive frames eye ratio above/below threshold
EYE_RATIO_CONSEC_FRAMES = 3

COUNTER = 0
# Keep track of number of blinks
TOTAL = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grabbing indexes of facial landmark for left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

while True:
    # if video is a file and no more frames to process
    if fileStream and not vs.more():
        break

    frame = vs.read() # read frame
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_locations = face_recognition.face_locations(frame)
    for face_location in face_locations:
        top = face_location[0]
        right = face_location[1]
        bottom = face_location[2]
        left = face_location[3]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        # No blinks detected yet
        if TOTAL == 0:
            cv2.putText(frame, "SPOOF", (bottom + 20, left),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "VERIFIED", (bottom + 20, left),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # detect faces in grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftRatio = eye_aspect_ratio(leftEye)
        rightRatio = eye_aspect_ratio(rightEye)

        ear = (leftRatio + rightRatio) / 2.0

        # Drawing eye shape on frame for visualisaiotn
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # ear is Eye Aspect Ratio
        # Every frame that this ratio is less than threshold, counter increase by one
        if ear < EYE_RATIO_THRESH:
            COUNTER += 1
        else:
            # If EAR less than threshold for more than or equals to 3 frames, one blink is registered
            if COUNTER >= EYE_RATIO_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Ratio: {:.25}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()