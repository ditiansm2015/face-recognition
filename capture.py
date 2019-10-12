import numpy as np
import cv2
import os

path='dataGenerated';
font=cv2.FONT_HERSHEY_SIMPLEX
i=0;

name=input('Enter your name: ')
name=name.title()
print('Hello',name)

folder=os.path.join(path,name);

if not os.path.exists(folder):
    os.mkdir(folder)

print('Created Folder with name ',name);

print('Initializing WebCam.........')

cap = cv2.VideoCapture(0)
cap.open(0)

print('Webcam Intialized\nUse key c to capture your image');

while(True):
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if i==10:
            print('Pictues Captured')
            break
    
        frame=cv2.flip(frame,1);
    
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(os.path.join(folder,str(i)+'.png'),frame)
            i=i+1

        text='Image Captured: '+str(i+1)
        cv2.putText(frame,text,(50,50),font,1,(0,0,255),1,cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as excp:
        continue
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
