import cv2
# Trimming off 4 frames in the front of one, and 4 in teh back of the other
doPole = False
if doPole:
    Path = 'C:\\Users\\CDSFSSkatter\\Documents\\Projects\\videotracking\\videotrackingrepo\\camera_calib_tools\\images\\axis-pole\\sampleFolder\\Mar10-2020-1538\\'
else:
    Path = 'C:\\Users\\CDSFSSkatter\\Documents\\Projects\\videotracking\\videotrackingrepo\\camera_calib_tools\\images\\axis-wall\\sampleFolder\\Mar10-2020-1538\\'

videoLink = Path + 'Mar10-2020-1538.mp4'

output = Path + 'newVid.mp4'

cap = cv2.VideoCapture(videoLink)

# count the frames
if 1:
    Count = 511
else:
    Count = 0
    while True:
        r, img = cap.read() 
        if r:
            Count += 1
        else:
            break
    cap.release()
    cap = cv2.VideoCapture(videoLink)

if doPole:
    beginFrame = 0
    endFrame = Count - 4
else:
    beginFrame = 4
    endFrame = Count

fps = cap.get(cv2.CAP_PROP_FPS)
imWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
imHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
#fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
fourcc = cv2.VideoWriter_fourcc('H', 'F', 'Y', 'U')

out = cv2.VideoWriter(output,fourcc, fps, (imWidth ,imHeight))

imCount = 0
while imCount < endFrame:
    r, img = cap.read()  
    if r:
        if imCount >= beginFrame:
            out.write(img)
        imCount += 1
    else:
        break
    print(imCount)
out.release()
cap.release()