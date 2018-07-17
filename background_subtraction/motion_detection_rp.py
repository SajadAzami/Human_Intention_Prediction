import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

print cv2.__version__


# returns the estimated position of the object using last n frames
def get_next_pos(measurements, n):
    measurements = np.array(measurements)
    if len(measurements) > n:
        x = measurements[-n:, 0]
        y = measurements[-n:, 1]
        x_average_loc = sum(x[-n:]) / n
        y_average_loc = sum(y[-n:]) / n
        x_speed = (x[-1] - x_average_loc) / 2
        y_speed = (y[-1] - y_average_loc) / 2
        # x_speed = (x[-1] - x[-n]) / 2
        # y_speed = (y[-1] - y[-n]) / 2
        return x[-1] + x_speed, y[-1] + y_speed
    else:
        x = measurements[-1][0]
        y = measurements[-1][1]
        return x, y


# checks if a rectangle contains a point
def rect_contains(rect, pt):
    logic = rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]
    return logic


# construct the argument parser and pa rse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

aoi_x1, aoi_y1, aoi_x2, aoi_y2 = 300, 300, 400, 450

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
door_locked = False
temp_measurement = []
door_status = False
lock_timer = 0

# loop over the frames of the video
while True:

    # door stabilizer, holds the door open for 50 frames
    if not door_locked:
        door_status = False
    if door_locked:
        lock_timer += 1
    if lock_timer > 50:
        lock_timer = 0
        door_locked = False

    (grabbed, frame) = camera.read()

    # if the frame could not be grabbed, then we have reached the end of the video
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

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, 5, 255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    counter = 0
    for c in cnts:
        # if the contour is too small or too big, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # predict motion using KalmanFilter
        temp_measurement.append([x + w / 2, y + h / 2])
        # print(temp_measurement[-1])
        # if len(temp_measurement) > 2:
        #     # print(temp_measurement[-2:])
        #     kf = KalmanFilter(initial_state_mean=0, n_dim_obs=2)
        #     # kf_results = kf.em(temp_measurement[-2:]).smooth(temp_measurement[-2:])
        #     # predictedCoords = (kf_results[0], kf_results[1])
        #     # print(predictedCoords)
        #     print((x + w) / 2, (y + h) / 2)

        # draw the predicted motion
        predicted_coords = get_next_pos(temp_measurement, 20)
        predicted_coords_long = get_next_pos(temp_measurement, 100)
        cv2.circle(frame, (predicted_coords[0], predicted_coords[1]), 2, (0, 255, 255), 3)
        cv2.line(frame, (x + w / 2, y + h / 2), (predicted_coords[0], predicted_coords[1]), (100, 10, 255), 2, 8)

        # draw a circle in the center of the bounding box
        cv2.circle(frame, (x + w / 2, y + h / 2), 2, (255, 0, 0), 3)

        # for real time door handler (using current location)
        # if rect_contains((aoi_x1, aoi_y1, aoi_x2, aoi_y2), (x + w / 2, y + h / 2)):
        if not door_locked:
            if rect_contains((aoi_x1, aoi_y1, aoi_x2, aoi_y2),
                             (predicted_coords[0], predicted_coords[1])):
                door_status = True
                door_locked = True
                counter += 1
            if counter == 0:
                door_status = False
    # draw the text and timestamp on the frame
    if door_status:
        print 'door open'
        # cv2.putText(frame, "Door Status: {}".format('Open'), (10, 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elif not door_status:
        print 'door close'
        # cv2.putText(frame, "Door Status: {}".format('Close'), (10, 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # define area of interest for the door
    cv2.rectangle(frame, (aoi_x1, aoi_y1), (aoi_x2, aoi_y2), (0, 100, 100), 2)

    # show the frame and record if the user presses a key
    # cv2.imshow("Door Handler", frame)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(10) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
