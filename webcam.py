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
    #out = grad_cam_live.live(gcam,frame)
    
  
    # Display the resulting frame
    #scale_percent = 250 # percent of original size
    #width = int(out.shape[1] * scale_percent*1.5 / 100)
    #height = int(out.shape[0] * scale_percent / 100)
    #dim = (width, height)
    # resize image
    #
    #out = cv2.resize(out, dim, interpolation = cv2.INTER_AREA)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Vesna"

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    textX = (frame.shape[1] - textsize[0]) // 2
    textY = (frame.shape[0] + textsize[1]) * 19 // 20

    # add text centered on image
    cv2.putText(frame, text, (textX, textY ), font, 1, (0, 0, 0), 2)
    frame = cv2.resize(frame,(200,200));
    images = [frame for i in range(5)]
     
    frame = np.concatenate(images, axis=1)
    frames = [frame for i in range(5)]
    frame = np.concatenate(frames, axis=0)
    cv2.imshow('frame',frame)
    #cv2.imshow('frame',frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
