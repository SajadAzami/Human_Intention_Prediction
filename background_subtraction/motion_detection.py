# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

print cv2.__version__


def rectContains(rect, pt):
    logic = rect[0] < pt[0] < rect[0] + rect[2] and rect[1] < pt[1] < rect[1] + rect[3]
    return logic


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(2, 2)

    def estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        print(predicted)
        return predicted


# construct the argument parser and pa rse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)

# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None
frames = []
avg = None

# loop over the frames of the video
kfObj = KalmanFilter()
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    door_status = False
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # # commented due to better implementation(dynamic first_frame)
    # # compute the absolute difference between the current frame and
    # # first frame
    # frameDelta = cv2.absdiff(firstFrame, gray)
    # thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # # dilate the thresholded image to fill in holes, then find contours
    # # on thresholded image
    # thresh = cv2.dilate(thresh, None, iterations=2)
    # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    #                              cv2.CHAIN_APPROX_SIMPLE)

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, 2, 255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # predict motion
        print((x + w) / 2, (y + h) / 2)
        predictedCoords = kfObj.estimate((x + w) / 2, (y + h) / 2)
        # draw the predicted motion
        cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 255, 255], 2, 8)
        cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
                 (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)

        # utility of the window
        text = ""
        if rectContains((80, 280, 400, 440), (x + (w / 2), y + (h / 2))):
            door_status = True

    # draw the text and timestamp on the frame
    if door_status:
        cv2.putText(frame, "Door Status: {}".format("Open"), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Door Status: {}".format("Close"), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # define area of interest for the door
    cv2.rectangle(frame, (80, 280), (400, 440), (100, 100, 100), 2)
    cv2.ellipse(frame, (240, 360), (160, 80), 0, 0, 360, (0, 0, 255), 2)

    # show the frame and record if the user presses a key
    cv2.imshow("Door Handler", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
