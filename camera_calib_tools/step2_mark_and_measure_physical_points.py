import numpy as np
import cv2
import video_channels
import select_camera_to_calibrate

config = video_channels.channel(select_camera_to_calibrate.Camera)
# make sure step2 is completed0
# keystrokes for zoom '+', unzoom '-', pan rigth 'r', pan left 'l', 'u', 'd'

cv2.namedWindow('cameraview',cv2.WINDOW_NORMAL)
zoomState = 1.0
zoomStep = 0.35
leftrightstate = 0
updownstate = 0
panStep = 50

xFractionMid = 0.5
if config.camName == 'warrior-pole':
    yFraction = 0.6 # bestr to have this at 0.5. as it would make the middle measurment onlyu dependent on one rot angle
    xFraction = 0.2
elif config.camName == 'warrior-wall':
    yFraction = 0.73 # bestr to have this at 0.5. as it would make the middle measurment onlyu dependent on one rot angle
    xFraction = 0.3
elif config.camName == 'axis-pole':
    yFraction = 0.65 # best to have this at 0.5. as it would make the middle measurment onlyu dependent on one rot angle
    xFraction = 0.2
    yFractionMid = 0.59
elif config.camName == 'axis-wall':
    yFraction = 0.6 # bestr to have this at 0.5. as it would make the middle measurment only dependent on one rot angle
    xFraction = 0.3
    yFractionMid = 0.5

n2 = config.imSize[0]
n1 = config.imSize[1]

yLine = int(n2 * yFraction)
drawPos = np.array([[n1*(0.5-xFraction),yLine],[n1/2,yLine],[n1*(0.5+xFraction),yLine],[n1*0.5,n2*yFractionMid],[n1/2,n2/2]]).astype(np.int)
drawColors = [(0,255,0),(0,0,255),(0,255,0),(0,0,255)]

np.savez(config.Folder+'drawPos',drawPos=drawPos)

cap = cv2.VideoCapture(config.CapturePath)

counter = 0
while(True):
    ret, frame = cap.read()
    # draw things in the frame now...
    if counter == 0:
        assert config.imSize[0] == frame.shape[0] and config.imSize[1] == frame.shape[1], "The frame is of a different size than what is specified in the configuration file (video_channels.py)"

    centerX = int(n1/2 + leftrightstate)
    centerV = int(n2/2 + updownstate)
    Left = max(0,int(centerX - n1/2/zoomState))
    Right = min(n1,int(centerX + n1/2/zoomState))
    Top = max(0,int(centerV - n2/2/zoomState))
    Bottom = min(n2,int(centerV + n2/2/zoomState))
    #oh, and write instructions too

    frame = frame[Top:Bottom,Left:Right]

    fontScale = int(np.ceil(0.6 *  frame.shape[0] / 1080))
    fthickness = int(np.ceil(3 *  frame.shape[0] / 1080))
    #print("font scale and thickness: ",fontScale,", ",fthickness)
    lineScale = int(5 * config.imSize[0] / 1080)
    lineWeight = int(1 * config.imSize[0] / 1080 )


    cv2.putText(frame,'Zoom and pan +,-,r, l, u, d. Mark the points with tape and measure horizontal distance to camera',(int(frame.shape[0]/15),int(frame.shape[0]/15)), cv2.FONT_ITALIC, fontScale,(0,0,255),fthickness)
    for i in range(len(drawColors)):
        cv2.circle(frame,(drawPos[i,0]-Left,drawPos[i,1]-Top), lineScale, drawColors[i], lineWeight,8)

    cv2.imshow('cameraview',frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('+'):
        zoomState += zoomStep
    elif key & 0xFF == ord('-'):
        if zoomState > 1.0:
            zoomState -= zoomStep
    elif key & 0xFF == ord('u'):
        updownstate -= panStep
    elif key & 0xFF == ord('d'):
        updownstate += panStep
    elif key & 0xFF == ord('l'):
        leftrightstate -= panStep
    elif key & 0xFF == ord('r'):
        leftrightstate += panStep
    counter += 1