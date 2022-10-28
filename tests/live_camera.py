import cv2
import video_channels

camName = 'axis-pole'
camName = 'axis-wall'
camName = 'hikvision-thermal'
myStream = video_channels.channel(camName)
cap = cv2.VideoCapture(myStream.CapturePath)
cv2.namedWindow('cameraview',cv2.WINDOW_NORMAL)
while(True):
    ret, frame = cap.read()
    print("im size: ",frame.shape)
    cv2.imshow('cameraview',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
cv2.imwrite('testIm.png',frame)