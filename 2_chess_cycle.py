import os
import time
from datetime import datetime
from picamera2 import Picamera2
import cv2
import numpy as np

# Photo session settings
total_photos = 30             # Number of images to take
countdown = 5                 # Interval for count-down timer, seconds
font = cv2.FONT_HERSHEY_SIMPLEX # Countdown timer font

# Camera settings
cam_width = 640               # Cam sensor width settings for each camera
cam_height = 480              # Cam sensor height settings

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

# Let's start taking photos!
counter = 0
t2 = datetime.now()
print("Starting photo sequence")

while counter < total_photos:
    frame1 = picam1.capture_array("main")
    frame2 = picam2.capture_array("main")

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

    #mirror both camera horizontally
    frame1 = cv2.flip(frame1, 1)
    frame2 = cv2.flip(frame2, 1)    
    
    # Combine frames side by side
    combined_frame = np.hstack((frame1, frame2))
    
    t1 = datetime.now()
    cntdwn_timer = countdown - int((t1-t2).total_seconds())
    
    # If countdown is zero - let's record next image
    if cntdwn_timer <= 0:
        counter += 1
        filename = f'./scenes/stereo_scene_{cam_width*2}x{cam_height}_{counter}.png'
        cv2.imwrite(filename, combined_frame)
        print(f' [{counter} of {total_photos}] {filename}')
        t2 = datetime.now()
        time.sleep(1)
        cntdwn_timer = countdown  # Reset the countdown
    
    # Draw countdown counter, seconds
    cv2.putText(combined_frame, str(max(0, cntdwn_timer)), (50,50), font, 2.0, (0,0,255), 4, cv2.LINE_AA)
    cv2.imshow("Stereo Pair", combined_frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'Q' key to quit, or wait till all photos are taken
    if key == ord("q"):
        break

print("Photo sequence finished")

picam1.stop()
picam2.stop()
cv2.destroyAllWindows()