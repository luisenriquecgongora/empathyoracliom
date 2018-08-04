####
import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascade_face.xml')

bufferCurrentTime = 0;

refocus = 10

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Focus Are Definition
    heightPort = gray.shape[0];
    widthPort = gray.shape[1];
    heightFocus = heightPort / 2
    widthFocus = widthPort / 4
    offsetLeft = (widthPort - widthFocus)/2
    offsetTop = (heightPort - heightFocus)/2
    focusArea = gray[offsetTop:offsetTop+heightFocus,offsetLeft:offsetLeft+widthFocus]
    cv2.rectangle(gray,(offsetLeft,offsetTop),(offsetLeft+widthFocus,offsetTop+heightFocus),(0,255,0),2)

    # Face detection on Focus Area
    faces = face_cascade.detectMultiScale(focusArea)
    for (x,y,w,h) in faces:
        #Display Face
        if (w>h):
            h = w
        else:
            w = h
        cv2.rectangle(gray,(offsetLeft+x,offsetTop+y),(offsetLeft+x+w,offsetTop+y+h),(255,0,0),2)
        roi_gray = gray[offsetTop+y:offsetTop+y+h, offsetLeft+x+refocus:offsetLeft+x+w-refocus]
        millis = int(round(time.time() * 1000))
        print(millis -  bufferCurrentTime)
        bufferCurrentTime = millis
        cv2.imshow('Face',roi_gray)
        #Border detection
        canniedImage = cv2.Canny(roi_gray,128,128)
        blurredImage = cv2.blur(canniedImage,(8,8))
        resized = cv2.resize(blurredImage, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('Canny',resized)
    # Display the Full Port
    cv2.imshow('Full',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

### Guardar imagen de rostro
### Leer entendimiento
### Guardar data
