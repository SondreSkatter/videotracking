import numpy as np
import video_channels
import camera
import calibrate_camera
import select_camera_to_calibrate

config = video_channels.channel(select_camera_to_calibrate.Camera)

# In the previous step, 4 measurements were made. Enter them in the following 4 lines
inchToM = 0.0254
# CAMERA POSITION IS DEFINED AS THE APERTURE POSITION. IT CAN BE HARD TO KNOW WHERE THAT IS. TYPICALLY IT WILL BE INSDIE OF THE FRONT LENSES, TOWARD  ThE BACK OF THE LENS MODULE
if config.camName == 'warrior-pole':
    camHeight = 83.5 * inchToM
    LeftDiag = (16*12+116-4.5-5.5) * inchToM
    midDiag = (16*12+103-6-5.5) * inchToM
    RightDiag = (16*12+105-5.5-5.5) * inchToM
    LeftDiag = (16*12+116-4.5-5.5 - 2.5) * inchToM
    midDiag = (16*12+103-6-5.5 - 2.5) * inchToM
    RightDiag = (16*12+105-5.5-5.5 - 2.5) * inchToM

    LeftToMiddle = 77.5 * inchToM
    RightToMiddle = 74.5 * inchToM
elif config.camName == 'warrior-wall':
    camHeight = 9 * 12 * inchToM
    LeftDiag = (16*12*2 + 7.75 - 8) * inchToM
    midDiag = (16*12*2 + 4*12 + 4.5 - 8) * inchToM
    RightDiag = (16*12*2 + 9. * 12 + 9.25 - 8) * inchToM

    LeftToMiddle = 114.75 * inchToM
    RightToMiddle = 137.5 * inchToM
elif config.camName == 'axis-wall':
    camHeight = (106.5) * inchToM
    LeftDiag = (264.5) * inchToM
    midDiag = (255.5) * inchToM
    RightDiag = (299.5) * inchToM
    LeftToMiddle = 124 * inchToM
    RightToMiddle = 137 * inchToM
    midToFar = 74 * inchToM
elif config.camName == 'axis-pole':
    camHeight = 85 * inchToM
    LeftDiag = 337.25 * inchToM
    midDiag = 327.75 * inchToM
    midToFar = 83.25 * inchToM
    RightDiag = 338 * inchToM
    LeftToMiddle = 101.2 * inchToM
    RightToMiddle = 100.75* inchToM

np.savez(config.Folder+'measuredDistances',camHeight=camHeight,LeftDiag=LeftDiag,midDiag=midDiag,RightDiag=RightDiag,LeftToMiddle=LeftToMiddle,RightToMiddle=RightToMiddle,midToFar=midToFar)

print("Looking good! Move on to step 3!")