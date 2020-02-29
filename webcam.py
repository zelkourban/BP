import numpy as np
import cv2
import grad_cam_live

cap = cv2.VideoCapture(0)

gcam = grad_cam_live.live_setup()
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = grad_cam_live.live(gcam,frame)
    
  
    # Display the resulting frame
    scale_percent = 250 # percent of original size
    width = int(out.shape[1] * scale_percent*1.5 / 100)
    height = int(out.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    #
    out = cv2.resize(out, dim, interpolation = cv2.INTER_AREA)
     
    cv2.imshow('frame',out)
    #cv2.imshow('frame',frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
