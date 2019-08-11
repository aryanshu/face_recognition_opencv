import cv2
import numpy as np
import pickle

face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye=cv2.CascadeClassifier('haarcascade_eye.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

label={"person_name":1}
with open("label_ids.pickle","rb") as f:
   label=pickle.load(f)
   label={v:k for k,v in label.items()}

frame1=cv2.imread('faces4.jpg')

frame=cv2.resize(frame1,(550,550),interpolation = cv2.INTER_AREA)

cv2.imshow('img',frame)
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

faces=face.detectMultiScale(gray,1.3,5)
    
for (x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    ngray=gray[y:y+h,x:x+w]
    ncolor=frame[y:y+h,x:x+w]
    #eyes=eye.detectMultiScale(ngray)

    id_,conf=recognizer.predict(ngray)
    #if conf>=30 and conf<=1100:
    print(id_)
    print(label[id_])
    print(conf)
    font=cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame,label[id_],(x,y),font,.4,(0,255,0))
    """for xi,yi,wi,hi in eyes:
        
        cv2.rectangle(ncolor,(xi,yi),(xi+wi,yi+hi),(0,255,0),2)"""

    cv2.imshow('img',frame)
    
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
