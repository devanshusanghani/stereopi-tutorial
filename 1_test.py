from picamera2 import Picamera2
import time
import cv2
import numpy as np
import os
from datetime import datetime

# File for captured image
filename = './scenes/stereo_photo.png'

# Camera settings
cam_width = 640  # Reduced for each camera
cam_height = 480

# Final image capture settings
scale_ratio = 1  # No scaling to maintain resolution

# Camera resolution height must be dividable by 16, and width by 32
cam_width = int((cam_width+31)/32)*32
cam_height = int((cam_height+15)/16)*16
print("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Initialize the cameras
picam1 = Picamera2(0)  # First camera
picam2 = Picamera2(1)  # Second camera

config1 = picam1.create_still_configuration(main={"size": (cam_width, cam_height), "format": "BGR888"})
config2 = picam2.create_still_configuration(main={"size": (cam_width, cam_height), "format": "BGR888"})

picam1.configure(config1)
picam2.configure(config2)

picam1.start()
picam2.start()

# Allow time for the cameras to warm up
time.sleep(2)

t2 = datetime.now()
counter = 0
avgtime = 0

# Capture frames from the cameras
while True:
    frame1 = picam1.capture_array("main")
    frame2 = picam2.capture_array("main")

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

    #mirror both camera horizontally
    frame1 = cv2.flip(frame1, 1)
    frame2 = cv2.flip(frame2, 1)    

    # Combine frames side by side
    combined_frame = np.hstack((frame1, frame2))
    
    counter += 1
    t1 = datetime.now()
    timediff = t1 - t2
    avgtime = avgtime + (timediff.total_seconds())
    
    cv2.imshow("Stereo Pair", combined_frame)
    key = cv2.waitKey(1) & 0xFF
    t2 = datetime.now()
    
    # if the `q` key was pressed, break from the loop and save last image
    if key == ord("q"):
        avgtime = avgtime / counter
        print("Average time between frames: " + str(avgtime))
        print("Average FPS: " + str(1/avgtime))
        if not os.path.isdir("./scenes"):
            os.makedirs("./scenes")
        cv2.imwrite(filename, combined_frame)
        break

picam1.stop()
picam2.stop()
cv2.destroyAllWindows()