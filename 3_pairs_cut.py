import cv2
import os

# Global variables preset
total_photos = 30
photo_width = 1280  # cam_width * 2 (640 * 2)
photo_height = 480  # cam_height (480)
img_width = 640     # Half of the total width (640)
img_height = 480    # Same as the height (480)
photo_counter = 0

# Main pair cut cycle
if not os.path.isdir("./pairs"):
    os.makedirs("./pairs")

while photo_counter != total_photos:
    photo_counter += 1
    filename = './scenes/stereo_scene_'+str(photo_width)+'x'+str(photo_height)+\
               '_'+str(photo_counter) + '.png'
    
    if not os.path.isfile(filename):
        print("No file named "+filename)
        continue

    pair_img = cv2.imread(filename, -1)
    
    cv2.imshow("ImagePair", pair_img)
    cv2.waitKey(0)
    
    # Split the stereo image into left and right
    imgLeft = pair_img[0:img_height, 0:img_width]  # Y+H and X+W
    imgRight = pair_img[0:img_height, img_width:photo_width]
    
    leftName = './pairs/left_'+str(photo_counter).zfill(2)+'.png'
    rightName = './pairs/right_'+str(photo_counter).zfill(2)+'.png'
    
    cv2.imwrite(leftName, imgLeft)
    cv2.imwrite(rightName, imgRight)
    
    print('Pair No '+str(photo_counter)+' saved.')

print('End cycle')