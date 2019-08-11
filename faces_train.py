import os
import numpy as np
import cv2
from PIL import Image
import pickle

BASE_DIR=os.path.dirname(os.path.realpath(__file__))
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img_path=os.path.join(BASE_DIR,'images')

recognizer=cv2.face.LBPHFaceRecognizer_create()
current_id=0
label_id={}
x_train=[]
y_label=[]

for root,dirs,files in os.walk(img_path):
   for file in files:
      if file.endswith("jpg"):
          path=os.path.join(root,file)
          label=os.path.basename(root)
          #print(label,path)
          
          if not label in label_id:
          	label_id[label]=current_id
          	current_id=current_id+1

          id_=label_id[label]
          #print(label_id)
          pil_image=Image.open(path).convert('L')
          pil_image2=pil_image.resize((550,550),Image.ANTIALIAS)
          numpy_img=np.array(pil_image2,"uint8")

          #print(numpy_img)
          faces=face.detectMultiScale(numpy_img)
          for (x,y,w,h) in faces:
             roi=numpy_img[y:y+h,x:x+w]
             x_train.append(roi)
             y_label.append(id_)
#print(y_label)
#print(x_train)

with open("label_ids.pickle","wb") as f:
	pickle.dump(label_id,f)

recognizer.train(x_train,np.array(y_label))
recognizer.save("model.yml")