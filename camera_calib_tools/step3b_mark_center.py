import numpy as np
import cv2
import video_channels

camName = 'warrior-pole'
myStream = video_channels.channel(camName)
# make sure step2 is completed0
# keystrokes for zoom '+', unzoom '-', pan rigth 'r', pan left 'l', 'u', 'd'

try:
    camParams = np.load(myStream.Folder+'camLensParams.npz')
except:
    print("step 2 needs to be completed!")
    exit

cv2.namedWindow('cameraview',cv2.WINDOW_NORMAL)
zoomState = 1.0
zoomStep = 0.25
leftrightstate = 0
updownstate = 0
panStep = 100


yFraction = 0.5 # bestr to have this at 0.5. as it would make the middle measurment onlyu dependent on one rot angle
xFraction = 0.05

yLine = int(camParams['n2'] * yFraction)
drawPos = np.array([[camParams['n1']*(0.5-xFraction),yLine],[camParams['n1']/2,yLine],[camParams['n1']*(0.5+xFraction),yLine]]).astype(np.int)
drawColors = [(0,255,0),(0,0,255),(0,255,0)]

cap = cv2.VideoCapture(myStream.CapturePath)

while(True):
    ret, frame = cap.read()
    # draw things in the frame now...

    centerX = int(camParams['n1']/2 + leftrightstate)
    centerV = int(camParams['n2']/2 + updownstate)
    Left = max(0,int(centerX - camParams['n1']/2/zoomState))
    Right = min(camParams['n1'],int(centerX + camParams['n1']/2/zoomState))
    Top = max(0,int(centerV - camParams['n2']/2/zoomState))
    Bottom = min(camParams['n2'],int(centerV + camParams['n2']/2/zoomState))
    #oh, and write instructions too

    frame = frame[Top:Bottom,Left:Right]
    cv2.putText(frame,'Zoom and pan +,-,r, l, u, d. Mark the points with tape and measure horizontal distance to camera',(Left+50,Top+50), cv2.FONT_ITALIC, 1,(0,0,255),3)
    for i in range(len(drawColors)):
        cv2.circle(frame,(drawPos[i,0]-Left,drawPos[i,1]-Top), 5, drawColors[i], 1,8)

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
        updownstate += panStep
    elif key & 0xFF == ord('d'):
        updownstate -= panStep
    elif key & 0xFF == ord('l'):
        leftrightstate -= panStep
    elif key & 0xFF == ord('r'):
        leftrightstate += panStep