from picamera2 import Picamera2
import time
import cv2
import numpy as np
import os
from datetime import datetime

# File for captured image
filename = './scenes/photo.png'

# Camera settings
cam_width = 1280
cam_height = 480

# Final image capture settings
scale_ratio = 0.5

# Camera resolution height must be dividable by 16, and width by 32
cam_width = int((cam_width+31)/32)*32
cam_height = int((cam_height+15)/16)*16
print("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Buffer for captured image settings
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
print("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": (cam_width, cam_height)},
    lores={"size": (img_width, img_height)},
    display="lores"
)
picam2.configure(config)
picam2.start()

# Allow time for the camera to warm up
time.sleep(2)

t2 = datetime.now()
counter = 0
avgtime = 0

# Capture frames from the camera
while True:
    frame = picam2.capture_array("lores")
    
    counter += 1
    t1 = datetime.now()
    timediff = t1 - t2
    avgtime = avgtime + (timediff.total_seconds())
    
    cv2.imshow("pair", frame)
    key = cv2.waitKey(1) & 0xFF
    t2 = datetime.now()
    
    # if the `q` key was pressed, break from the loop and save last image
    if key == ord("q"):
        avgtime = avgtime / counter
        print("Average time between frames: " + str(avgtime))
        print("Average FPS: " + str(1/avgtime))
        if not os.path.isdir("./scenes"):
            os.makedirs("./scenes")
        cv2.imwrite(filename, frame)
        break

picam2.stop()
cv2.destroyAllWindows()