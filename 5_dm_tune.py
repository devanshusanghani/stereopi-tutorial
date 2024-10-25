import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import json
from stereovision.calibration import StereoCalibration
from picamera2 import Picamera2
import time

# Global variables preset
photo_width = 1280
photo_height = 480
img_width = 640
img_height = 480
image_size = (img_width, img_height)

# Stereo parameters with initial values
SWS = 5
PFS = 5
PFC = 29
MDS = -25
NOD = 128
TTH = 100
UR = 10
SR = 15
SPWS = 100
loading_settings = 0

def initialize_cameras():
    # Get list of camera devices
    cam_list = Picamera2.global_camera_info()
    if len(cam_list) < 2:
        raise RuntimeError("Not enough cameras detected! Need 2 cameras.")
    
    # Initialize cameras with specific device IDs
    picam2_left = Picamera2(0)
    picam2_right = Picamera2(1)
    
    # Set up camera configurations
    config_left = picam2_left.create_still_configuration(
        main={"size": (img_width, img_height), "format": "BGR888"}
    )
    config_right = picam2_right.create_still_configuration(
        main={"size": (img_width, img_height), "format": "BGR888"}
    )
    
    # Configure and start cameras
    picam2_left.configure(config_left)
    picam2_right.configure(config_right)
    time.sleep(0.5)
    picam2_left.start()
    picam2_right.start()
    
    return picam2_left, picam2_right

def capture_stereo_images(picam2_left, picam2_right):
    time.sleep(0.5)
    img_left = picam2_left.capture_array()
    time.sleep(0.1)
    img_right = picam2_right.capture_array()
    
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    return img_left_gray, img_right_gray

def stereo_depth_map(rectified_pair):
    print(f'SWS={SWS} PFS={PFS} PFC={PFC} MDS={MDS} NOD={NOD} TTH={TTH}')
    print(f'UR={UR} SR={SR} SPWS={SPWS}')
    
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    
    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    
    disparity = sbm.compute(dmLeft, dmRight)
    
    local_max = disparity.max()
    local_min = disparity.min()
    print(f"MAX {local_max} MIN {local_min}")
    
    if local_max - local_min == 0:
        return np.zeros_like(disparity, dtype=np.float32)
    
    disparity_visual = (disparity - local_min) * (1.0 / (local_max - local_min))
    return disparity_visual

def save_map_settings(event):
    buttons.label.set_text("Saving...")
    print('Saving to file...')
    result = json.dumps({
        'SADWindowSize': SWS,
        'preFilterSize': PFS,
        'preFilterCap': PFC,
        'minDisparity': MDS,
        'numberOfDisparities': NOD,
        'textureThreshold': TTH,
        'uniquenessRatio': UR,
        'speckleRange': SR,
        'speckleWindowSize': SPWS
    }, sort_keys=True, indent=4, separators=(',', ':'))
    
    fName = '3dmap_set.txt'
    with open(str(fName), 'w') as f:
        f.write(result)
    
    buttons.label.set_text("Save to file")
    print(f'Settings saved to file {fName}')

def load_map_settings(event):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    loading_settings = 1
    fName = '3dmap_set.txt'
    print('Loading parameters from file...')
    buttonl.label.set_text("Loading...")
    
    with open(fName, 'r') as f:
        data = json.load(f)
        sSWS.set_val(data['SADWindowSize'])
        sPFS.set_val(data['preFilterSize'])
        sPFC.set_val(data['preFilterCap'])
        sMDS.set_val(data['minDisparity'])
        sNOD.set_val(data['numberOfDisparities'])
        sTTH.set_val(data['textureThreshold'])
        sUR.set_val(data['uniquenessRatio'])
        sSR.set_val(data['speckleRange'])
        sSPWS.set_val(data['speckleWindowSize'])
    
    buttonl.label.set_text("Load settings")
    print(f'Parameters loaded from file {fName}')
    print('Redrawing depth map with loaded parameters...')
    loading_settings = 0
    update(0)
    print('Done!')

def update(val):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
    SWS = int(sSWS.val/2)*2+1
    PFS = int(sPFS.val/2)*2+1
    PFC = int(sPFC.val/2)*2+1
    MDS = int(sMDS.val)
    NOD = int(sNOD.val/16)*16
    TTH = int(sTTH.val)
    UR = int(sUR.val)
    SR = int(sSR.val)
    SPWS = int(sSPWS.val)
    
    if loading_settings == 0:
        print('Rebuilding depth map')
        disparity = stereo_depth_map(rectified_pair)
        dmObject.set_data(disparity)
        print('Redraw depth map')
        plt.draw()

def main():
    global rectified_pair, buttons, buttonl, sSWS, sPFS, sPFC, sMDS, sNOD, sTTH, sUR, sSR, sSPWS, dmObject
    
    try:
        # Initialize cameras
        picam2_left, picam2_right = initialize_cameras()
        
        # Capture stereo pair
        img_left, img_right = capture_stereo_images(picam2_left, picam2_right)
        
        # Rectify images
        calibration = StereoCalibration(input_folder='calib_result')
        rectified_pair = calibration.rectify((img_left, img_right))
        
        if rectified_pair[0] is None:
            raise RuntimeError("Failed to rectify images")
        
        # Calculate initial disparity
        disparity = stereo_depth_map(rectified_pair)
        
        # Set up the UI
        axcolor = 'lightgoldenrodyellow'
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.subplots_adjust(left=0.15, bottom=0.5)
        
        # Show left image and depth map
        ax1.imshow(rectified_pair[0], 'gray')
        dmObject = ax2.imshow(disparity, aspect='equal', cmap='jet')
        
        # Create sliders
        slider_axes = []
        for i in range(9):
            ax = plt.axes([0.15, 0.01 + i*0.04, 0.7, 0.025], facecolor=axcolor)
            slider_axes.append(ax)
        
        sSWS = Slider(slider_axes[0], 'SWS', 5.0, 255.0, valinit=5)
        sPFS = Slider(slider_axes[1], 'PFS', 5.0, 255.0, valinit=5)
        sPFC = Slider(slider_axes[2], 'PreFiltCap', 5.0, 63.0, valinit=29)
        sMDS = Slider(slider_axes[3], 'MinDISP', -100.0, 100.0, valinit=-25)
        sNOD = Slider(slider_axes[4], 'NumOfDisp', 16.0, 256.0, valinit=128)
        sTTH = Slider(slider_axes[5], 'TxtrThrshld', 0.0, 1000.0, valinit=100)
        sUR = Slider(slider_axes[6], 'UnicRatio', 1.0, 20.0, valinit=10)
        sSR = Slider(slider_axes[7], 'SpcklRng', 0.0, 40.0, valinit=15)
        sSPWS = Slider(slider_axes[8], 'SpklWinSze', 0.0, 300.0, valinit=100)
        
        # Create buttons
        saveax = plt.axes([0.3, 0.38, 0.15, 0.04])
        buttons = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')
        buttons.on_clicked(save_map_settings)
        
        loadax = plt.axes([0.5, 0.38, 0.15, 0.04])
        buttonl = Button(loadax, 'Load settings', color=axcolor, hovercolor='0.975')
        buttonl.on_clicked(load_map_settings)
        
        # Connect sliders to update function
        for slider in [sSWS, sPFS, sPFC, sMDS, sNOD, sTTH, sUR, sSR, sSPWS]:
            slider.on_changed(update)
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        if 'picam2_left' in locals():
            picam2_left.stop()
        if 'picam2_right' in locals():
            picam2_right.stop()

if __name__ == "__main__":
    main()