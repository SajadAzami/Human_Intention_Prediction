# Pedestrian Trajectory Prediction on RaspberryPi
My B.Sc. project at Amirkabir University of Technology, Advisor: Dr. Shiry


This is a simple system for implementation of an automatic door using RGB data and a Raspberry Pi 3 (or any other device running Python and OpenCV).

OpenCV Version: 3.4.1
Python Version: 2.7 

The system first performs a Motion Detection using a simple (yet effective) background subtraction. Then naive estimations of the trajectories for moving objects performed and the door status is classified.

Using USB camera:
```
python motion_detection.py
```

Using recorded video:
```
python motion_detection.py	--video ../data/1_trimmed_1.mp4
```

For Raspberry Pi:
```
python motion_detection_rp.py	
```

add ```--min-area``` to set the threshold for minimum contour area of detected objects. 1200 is recommended for examples in _data_ folder.

```aoi_x1, aoi_y1, aoi_x2, aoi_y2``` are parameters for setting (x,y) location for top left and bottom right of Area of Interest (which is the door).
